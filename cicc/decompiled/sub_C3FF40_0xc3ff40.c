// Function: sub_C3FF40
// Address: 0xc3ff40
//
__int64 __fastcall sub_C3FF40(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        _BYTE *a7)
{
  _DWORD *v9; // rax
  _DWORD *v10; // rbx
  unsigned int v11; // r13d
  _QWORD *i; // r12
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-58h]
  _DWORD *v18; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v19; // [rsp+28h] [rbp-48h]

  sub_C3E660((__int64)&v16, a1);
  v9 = sub_C33340();
  v10 = v9;
  if ( v9 == dword_3F65580 )
    sub_C3C640(&v18, (__int64)v9, &v16);
  else
    sub_C3B160((__int64)&v18, dword_3F65580, &v16);
  if ( v18 == v10 )
    v11 = sub_C3FF40((unsigned int)&v18, (_DWORD)a2, a3, a4, a5, a6, (__int64)a7);
  else
    v11 = sub_C34710((__int64)&v18, a2, a3, a4, a5, a6, a7);
  if ( v18 == v10 )
  {
    if ( v19 )
    {
      for ( i = &v19[3 * *(v19 - 1)]; v19 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v10 == (_DWORD *)*i )
            break;
          sub_C338F0((__int64)i);
          if ( v19 == i )
            goto LABEL_13;
        }
      }
LABEL_13:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v18);
  }
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return v11;
}
