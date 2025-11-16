// Function: sub_14ABE30
// Address: 0x14abe30
//
__int64 __fastcall sub_14ABE30(unsigned __int8 *a1)
{
  unsigned __int8 *v1; // r12
  int v2; // eax
  char v3; // al
  __int64 v4; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  int v10; // r15d
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-38h]

  v1 = a1;
  if ( (unsigned __int8)sub_1642F90(*(_QWORD *)a1, 8) )
    return (__int64)a1;
  v2 = a1[16];
  if ( (unsigned __int8)v2 > 0x10u )
    goto LABEL_10;
  if ( !(unsigned __int8)sub_1593BB0(a1) )
  {
    v2 = a1[16];
    if ( (_BYTE)v2 != 14 )
      goto LABEL_10;
    v3 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    if ( v3 == 2 )
    {
      v14 = sub_16498A0(a1);
      v15 = sub_1643350(v14);
      a1 = (unsigned __int8 *)sub_15A4510(a1, v15, 0);
      if ( *(_BYTE *)(*(_QWORD *)v1 + 8LL) != 3 )
      {
        v2 = a1[16];
        v1 = a1;
LABEL_10:
        if ( (_BYTE)v2 == 13 )
        {
          if ( (v1[32] & 7) == 0 && (unsigned __int8)sub_16A8E60(v1 + 24, 8) )
          {
            sub_16A5A50(&v16, v1 + 24);
            v8 = sub_16498A0(v1);
            v4 = sub_159C0E0(v8, &v16);
            if ( v17 > 0x40 )
            {
              if ( v16 )
                j_j___libc_free_0_0(v16);
            }
            return v4;
          }
        }
        else if ( (unsigned int)(v2 - 11) <= 1 )
        {
          v9 = sub_15A0940(v1, 0);
          v4 = sub_14ABE30(v9);
          if ( v4 )
          {
            v10 = sub_15958F0(v1);
            if ( v10 == 1 )
              return v4;
            v11 = 1;
            while ( v9 == sub_15A0940(v1, v11) )
            {
              if ( v10 == ++v11 )
                return v4;
            }
          }
        }
        return 0;
      }
    }
    else if ( v3 != 3 )
    {
      return 0;
    }
    v6 = sub_16498A0(a1);
    v7 = sub_1643360(v6);
    v1 = (unsigned __int8 *)sub_15A4510(v1, v7, 0);
    v2 = v1[16];
    goto LABEL_10;
  }
  v12 = sub_16498A0(a1);
  v13 = sub_1643330(v12);
  return sub_15A06D0(v13);
}
