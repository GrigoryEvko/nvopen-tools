// Function: sub_2B2EA50
// Address: 0x2b2ea50
//
__int64 __fastcall sub_2B2EA50(_QWORD *a1, __int64 a2, unsigned __int16 a3)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 *v7; // r14
  unsigned __int64 v8; // rax
  char v9; // al
  __int64 v10; // rdi
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r15
  unsigned int v15; // ebx
  __int64 v17; // rdx
  int v18; // r12d
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  char v23; // [rsp+7h] [rbp-F9h]
  __int64 v24; // [rsp+8h] [rbp-F8h]
  unsigned int v25; // [rsp+8h] [rbp-F8h]
  __int64 v26; // [rsp+18h] [rbp-E8h]
  _BYTE v27[32]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v28; // [rsp+40h] [rbp-C0h]
  _BYTE v29[32]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v30; // [rsp+70h] [rbp-90h]
  __m128i v31; // [rsp+80h] [rbp-80h] BYREF
  __int64 v32; // [rsp+90h] [rbp-70h]
  __int64 v33; // [rsp+98h] [rbp-68h]
  __int64 v34; // [rsp+A0h] [rbp-60h]
  __int64 v35; // [rsp+A8h] [rbp-58h]
  __int64 v36; // [rsp+B0h] [rbp-50h]
  __int64 v37; // [rsp+B8h] [rbp-48h]
  __int16 v38; // [rsp+C0h] [rbp-40h]

  v4 = *a1;
  if ( (unsigned int)*(unsigned __int8 *)(*a1 + 8LL) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  v24 = *(_QWORD *)(a2 + 8);
  v5 = a2;
  if ( *(_QWORD *)(v24 + 24) != v4 )
  {
    v6 = a1[15];
    v28 = 257;
    v7 = (__int64 *)a1[14];
    v8 = *(_QWORD *)(v6 + 3344);
    v38 = 257;
    v31 = (__m128i)v8;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v9 = sub_9AC470(a2, &v31, 0);
    v10 = *a1;
    v11 = v9 ^ 1;
    if ( HIBYTE(a3) )
      v11 = a3;
    v23 = v11;
    if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
      v10 = **(_QWORD **)(v10 + 16);
    LODWORD(v26) = *(_DWORD *)(v24 + 32);
    BYTE4(v26) = *(_BYTE *)(v24 + 8) == 18;
    v12 = sub_BCE1B0((__int64 *)v10, v26);
    v13 = *(_QWORD *)(a2 + 8);
    v14 = v12;
    v25 = sub_BCB060(v13);
    v15 = 39 - ((v23 == 0) - 1);
    if ( v25 > (unsigned int)sub_BCB060(v14) )
      v15 = 38;
    if ( v14 == v13 )
    {
      return a2;
    }
    else
    {
      v5 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v7[10] + 120LL))(
             v7[10],
             v15,
             a2,
             v14);
      if ( !v5 )
      {
        v30 = 257;
        v5 = sub_B51D30(v15, a2, v14, (__int64)v29, 0, 0);
        if ( (unsigned __int8)sub_920620(v5) )
        {
          v17 = v7[12];
          v18 = *((_DWORD *)v7 + 26);
          if ( v17 )
            sub_B99FD0(v5, 3u, v17);
          sub_B45150(v5, v18);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v7[11] + 16LL))(
          v7[11],
          v5,
          v27,
          v7[7],
          v7[8]);
        v19 = *v7;
        v20 = *v7 + 16LL * *((unsigned int *)v7 + 2);
        if ( *v7 != v20 )
        {
          do
          {
            v21 = *(_QWORD *)(v19 + 8);
            v22 = *(_DWORD *)v19;
            v19 += 16;
            sub_B99FD0(v5, v22, v21);
          }
          while ( v20 != v19 );
        }
      }
    }
  }
  return v5;
}
