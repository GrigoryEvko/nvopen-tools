// Function: sub_1E74440
// Address: 0x1e74440
//
unsigned __int16 *__fastcall sub_1E74440(__int64 *a1, __int64 a2, __int64 *a3, int a4, __int64 a5)
{
  unsigned __int16 *result; // rax
  __int64 *v6; // r12
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  int v13; // ecx
  __int64 v14; // rcx
  char v15; // dl
  __int64 *v16; // [rsp+8h] [rbp-78h]
  __int64 *v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  int v19; // [rsp+28h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int128 v21; // [rsp+38h] [rbp-48h]
  __int64 v22; // [rsp+48h] [rbp-38h]

  result = *(unsigned __int16 **)(a2 + 72);
  v6 = *(__int64 **)(a2 + 64);
  v16 = (__int64 *)result;
  if ( v6 != (__int64 *)result )
  {
    do
    {
      v12 = *v6;
      v17 = a3;
      v18 = *a3;
      v13 = *((_DWORD *)a3 + 2);
      v21 = 0u;
      v19 = v13;
      LOBYTE(v13) = *(_DWORD *)(a2 + 24) == 1;
      v22 = 0;
      v20 = 0;
      sub_1E74370((__int64)a1, (__int64)&v18, v12, v13, a4, a4);
      v14 = 0;
      if ( *(_BYTE *)(a5 + 25) == BYTE1(v21) )
        v14 = a2;
      result = (unsigned __int16 *)(*(__int64 (__fastcall **)(__int64 *, __int64, __int64 *, __int64))(*a1 + 136))(
                                     a1,
                                     a5,
                                     &v18,
                                     v14);
      v15 = v21;
      a3 = v17;
      if ( (_BYTE)v21 )
      {
        if ( !v22 )
        {
          result = sub_1E736C0((__int64)&v18, a1[16], a1[2]);
          v15 = v21;
          a3 = v17;
        }
        *(_BYTE *)(a5 + 24) = v15;
        v10 = v20;
        *(_BYTE *)(a5 + 25) = BYTE1(v21);
        v11 = *(_QWORD *)((char *)&v21 + 2);
        *(_QWORD *)(a5 + 16) = v10;
        *(_QWORD *)(a5 + 26) = v11;
        *(_DWORD *)(a5 + 34) = *(_DWORD *)((char *)&v21 + 10);
        *(_QWORD *)(a5 + 40) = v22;
      }
      ++v6;
    }
    while ( v16 != v6 );
  }
  return result;
}
