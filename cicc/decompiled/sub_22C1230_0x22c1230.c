// Function: sub_22C1230
// Address: 0x22c1230
//
__int64 __fastcall sub_22C1230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // rdi
  unsigned __int8 *v8; // rax
  int v9; // ecx
  unsigned __int8 *v10; // rax
  int v11; // ecx
  unsigned __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]
  unsigned __int64 v16; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-28h]

  if ( a3 != *(_QWORD *)(a4 - 32) )
  {
    *(_WORD *)a1 = 6;
    return a1;
  }
  v7 = *(_QWORD *)(a3 + 8);
  if ( (*(_BYTE *)(a4 + 1) & 2) != 0 )
  {
    if ( a5 )
    {
      v8 = (unsigned __int8 *)sub_AD64C0(v7, 1, 0);
      *(_WORD *)a1 = 0;
      v9 = *v8;
      if ( (unsigned int)(v9 - 12) > 1 )
      {
        if ( (_BYTE)v9 == 17 )
        {
          v13 = *((_DWORD *)v8 + 8);
          if ( v13 > 0x40 )
            sub_C43780((__int64)&v12, (const void **)v8 + 3);
          else
            v12 = *((_QWORD *)v8 + 3);
          sub_AADBC0((__int64)&v14, (__int64 *)&v12);
          sub_22C00F0(a1, (__int64)&v14, 0, 0, 1u);
          if ( v17 > 0x40 && v16 )
            j_j___libc_free_0_0(v16);
          if ( v15 > 0x40 && v14 )
            j_j___libc_free_0_0(v14);
          if ( v13 > 0x40 )
          {
            if ( v12 )
              j_j___libc_free_0_0(v12);
          }
          return a1;
        }
LABEL_25:
        *(_BYTE *)a1 = 2;
        *(_QWORD *)(a1 + 8) = v8;
        return a1;
      }
    }
    else
    {
      v8 = (unsigned __int8 *)sub_AD6530(v7, a2);
      *(_WORD *)a1 = 0;
      v11 = *v8;
      if ( (unsigned int)(v11 - 12) > 1 )
      {
        if ( (_BYTE)v11 == 17 )
        {
          v13 = *((_DWORD *)v8 + 8);
          if ( v13 > 0x40 )
            sub_C43780((__int64)&v12, (const void **)v8 + 3);
          else
            v12 = *((_QWORD *)v8 + 3);
          sub_AADBC0((__int64)&v14, (__int64 *)&v12);
          sub_22C00F0(a1, (__int64)&v14, 0, 0, 1u);
          sub_969240((__int64 *)&v16);
          sub_969240((__int64 *)&v14);
          sub_969240((__int64 *)&v12);
          return a1;
        }
        goto LABEL_25;
      }
    }
    *(_BYTE *)a1 = 1;
    return a1;
  }
  else
  {
    if ( a5 )
      v10 = (unsigned __int8 *)sub_AD6530(v7, a2);
    else
      v10 = (unsigned __int8 *)sub_AD62B0(v7);
    *(_WORD *)a1 = 0;
    sub_22C0430(a1, v10);
    return a1;
  }
}
