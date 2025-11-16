// Function: sub_13EA060
// Address: 0x13ea060
//
int *__fastcall sub_13EA060(int *a1, __int64 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // eax
  bool v5; // zf
  unsigned int v6; // eax
  int v8; // eax
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  bool v13; // cc
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-28h]
  __int64 v18; // [rsp+10h] [rbp-20h]
  unsigned int v19; // [rsp+18h] [rbp-18h]

  *a1 = 0;
  v3 = *((_DWORD *)a2 + 2);
  *((_DWORD *)a2 + 2) = 0;
  v17 = v3;
  v16 = *a2;
  v4 = *((_DWORD *)a2 + 6);
  *((_DWORD *)a2 + 6) = 0;
  v5 = *a1 == 3;
  v19 = v4;
  v18 = a2[2];
  if ( !v5 )
  {
    if ( !(unsigned __int8)sub_158A120(&v16) )
    {
      v6 = v17;
      *a1 = 3;
      a1[4] = v6;
      *((_QWORD *)a1 + 1) = v16;
      a1[8] = v19;
      *((_QWORD *)a1 + 3) = v18;
      return a1;
    }
    goto LABEL_6;
  }
  if ( (unsigned __int8)sub_158A120(&v16) )
  {
LABEL_6:
    v8 = *a1;
    if ( *a1 != 4 )
    {
      if ( (unsigned int)(v8 - 1) > 1 )
      {
        if ( v8 == 3 )
        {
          if ( (unsigned int)a1[8] > 0x40 )
          {
            v10 = *((_QWORD *)a1 + 3);
            if ( v10 )
              j_j___libc_free_0_0(v10);
          }
          if ( (unsigned int)a1[4] > 0x40 )
          {
            v11 = *((_QWORD *)a1 + 1);
            if ( v11 )
              j_j___libc_free_0_0(v11);
          }
        }
      }
      else
      {
        *((_QWORD *)a1 + 1) = 0;
      }
      *a1 = 4;
    }
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    v9 = v17;
    goto LABEL_14;
  }
  if ( (unsigned int)a1[4] > 0x40 )
  {
    v12 = *((_QWORD *)a1 + 1);
    if ( v12 )
      j_j___libc_free_0_0(v12);
  }
  v13 = (unsigned int)a1[8] <= 0x40;
  *((_QWORD *)a1 + 1) = v16;
  v14 = v17;
  v17 = 0;
  a1[4] = v14;
  if ( v13 || (v15 = *((_QWORD *)a1 + 3)) == 0 )
  {
    *((_QWORD *)a1 + 3) = v18;
    a1[8] = v19;
    return a1;
  }
  j_j___libc_free_0_0(v15);
  v9 = v17;
  *((_QWORD *)a1 + 3) = v18;
  a1[8] = v19;
LABEL_14:
  if ( v9 <= 0x40 || !v16 )
    return a1;
  j_j___libc_free_0_0(v16);
  return a1;
}
