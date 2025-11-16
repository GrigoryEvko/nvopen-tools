// Function: sub_2ECA350
// Address: 0x2eca350
//
__int64 __fastcall sub_2ECA350(_QWORD *a1, __int64 a2, __int64 a3, int a4, __int64 a5)
{
  unsigned int v5; // r12d
  bool v6; // zf
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // r14
  __int64 v12; // rdx
  int v15; // [rsp+14h] [rbp-5Ch]
  _QWORD *v16; // [rsp+18h] [rbp-58h]
  int v17; // [rsp+28h] [rbp-48h] BYREF
  int v18; // [rsp+2Ch] [rbp-44h] BYREF
  int v19; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+34h] [rbp-3Ch] BYREF
  int v21; // [rsp+38h] [rbp-38h] BYREF
  _DWORD v22[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = a2;
  v16 = a1;
  v15 = a3;
  if ( (_DWORD)a3 != a4 )
  {
    v6 = *(_QWORD *)(a5 + 16) == 0;
    v17 = a2;
    v18 = a3;
    v19 = a4;
    if ( v6 )
LABEL_9:
      sub_4263D6(a1, a2, a3);
    v8 = (*(__int64 (__fastcall **)(__int64, int *, int *, int *))(a5 + 24))(a5, &v17, &v18, &v19);
    v10 = v9;
    v11 = (_QWORD *)*a1;
    if ( (_QWORD *)*a1 != a1 )
    {
      do
      {
        a1 = (_QWORD *)v8;
        a2 = v10;
        if ( sub_2ECA300(v8, v10, v11[2], v11[3]) )
        {
          v5 = *((_DWORD *)v11 + 6) + v5 - v8;
          v6 = *(_QWORD *)(a5 + 16) == 0;
          v21 = v15;
          v20 = v5;
          v22[0] = a4;
          if ( v6 )
            goto LABEL_9;
          v8 = (*(__int64 (__fastcall **)(__int64, unsigned int *, int *, _DWORD *))(a5 + 24))(a5, &v20, &v21, v22);
          v10 = v12;
        }
        v11 = (_QWORD *)*v11;
      }
      while ( v16 != v11 );
    }
  }
  return v5;
}
