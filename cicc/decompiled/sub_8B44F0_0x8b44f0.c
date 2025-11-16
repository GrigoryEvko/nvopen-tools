// Function: sub_8B44F0
// Address: 0x8b44f0
//
__int64 __fastcall sub_8B44F0(__int64 *a1, _QWORD *a2, __m128i *a3, __int64 a4, __int64 *a5, int *a6, __m128i *a7)
{
  _QWORD *v7; // r12
  __int64 *v8; // rbx
  char v12; // al
  __int64 v14; // rdx
  bool v15; // dl
  _BOOL4 v16; // eax
  int v17; // eax
  __int64 **v18; // rax
  __int64 v19; // rsi
  int v20; // eax
  int *v21; // [rsp-60h] [rbp-60h]
  bool v22; // [rsp-60h] [rbp-60h]
  int *v23; // [rsp-58h] [rbp-58h]
  __int64 **v24; // [rsp-58h] [rbp-58h]
  int *v25; // [rsp-58h] [rbp-58h]
  __int64 *v26; // [rsp-40h] [rbp-40h] BYREF

  if ( !a1 )
    return 1;
  v7 = a2;
  if ( !a2 )
    return 1;
  v8 = a1;
  while ( 1 )
  {
    v12 = *(_BYTE *)(v8[1] + 80);
    if ( v12 != *(_BYTE *)(v7[1] + 80LL) )
      return 0;
    if ( v12 == 3 )
      goto LABEL_16;
    v14 = v7[8];
    if ( v12 == 2 )
    {
      v25 = a6;
      v18 = sub_8A2270(*(_QWORD *)(v14 + 128), a3, a4, a5, 0, a6, a7);
      if ( *v25 )
        return 0;
      if ( (*((_BYTE *)v8 + 57) & 8) != 0 )
      {
        v19 = *(_QWORD *)(v8[8] + 128);
        v26 = 0;
        v22 = (unsigned int)sub_8B3500((__m128i *)v18, v19, (__int64 *)&v26, a4, 0) == 0;
        sub_725130(v26);
        v16 = v22;
        v15 = v22;
        a6 = v25;
        goto LABEL_11;
      }
      v21 = v25;
      v24 = v18;
      v17 = sub_8D3EA0(v18);
      a6 = v21;
      if ( !v17 )
      {
        v20 = sub_8DED30(*(_QWORD *)(v8[8] + 128), v24, 4096);
        a6 = v21;
        v15 = v20 == 0;
        v16 = v20 == 0;
        goto LABEL_11;
      }
LABEL_16:
      v15 = 0;
      v16 = 0;
      goto LABEL_11;
    }
    v23 = a6;
    if ( !(unsigned int)sub_8B44F0(**(_QWORD **)(v8[8] + 32), **(_QWORD **)(v14 + 32), a3, a4, a5, a6, a7) )
      return 0;
    a6 = v23;
    v15 = *v23 != 0;
    v16 = v15;
LABEL_11:
    v8 = (__int64 *)*v8;
    v7 = (_QWORD *)*v7;
    if ( !v8 || v15 )
      return !v16;
    if ( !v7 )
      return 1;
  }
}
