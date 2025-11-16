// Function: sub_2A63320
// Address: 0x2a63320
//
__int64 __fastcall sub_2A63320(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  char v8; // dl
  int v9; // esi
  char v11; // bl
  bool v12; // bl
  unsigned __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-58h]
  unsigned __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-48h]
  unsigned __int64 v17; // [rsp+30h] [rbp-40h]
  unsigned int v18; // [rsp+38h] [rbp-38h]

  v8 = *(_BYTE *)a2;
  v9 = *a4;
  if ( (unsigned int)(v9 - 12) > 1 )
  {
    if ( v8 != 2 )
    {
      if ( (_BYTE)v9 != 17 )
      {
        *(_BYTE *)a2 = 2;
        *(_QWORD *)(a2 + 8) = a4;
        goto LABEL_4;
      }
      v11 = a5;
      v14 = *((_DWORD *)a4 + 8);
      if ( v14 > 0x40 )
        sub_C43780((__int64)&v13, (const void **)a4 + 3);
      else
        v13 = *((_QWORD *)a4 + 3);
      sub_AADBC0((__int64)&v15, (__int64 *)&v13);
      v12 = sub_2A62120((char *)a2, (__int64)&v15, v11, 0, 1u);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      if ( v14 > 0x40 )
      {
        if ( v13 )
          j_j___libc_free_0_0(v13);
      }
      if ( v12 )
        goto LABEL_4;
    }
  }
  else if ( v8 != 1 )
  {
    *(_BYTE *)a2 = 1;
LABEL_4:
    sub_2A62F90(a1, (_BYTE *)a2, a3, (__int64)a4, a5, a6);
    return 1;
  }
  return 0;
}
