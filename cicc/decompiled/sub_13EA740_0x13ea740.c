// Function: sub_13EA740
// Address: 0x13ea740
//
__int64 __fastcall sub_13EA740(int *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v4; // eax
  int v5; // eax
  __int64 v6; // rdi
  bool v7; // cc
  unsigned int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]
  __int64 v16; // [rsp+20h] [rbp-30h]
  unsigned int v17; // [rsp+28h] [rbp-28h]

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)result == 13 )
  {
    v13 = *(_DWORD *)(a2 + 32);
    if ( v13 > 0x40 )
      sub_16A4FD0(&v12, a2 + 24);
    else
      v12 = *(_QWORD *)(a2 + 24);
    sub_1589870(&v14, &v12);
    if ( *a1 == 3 )
    {
      if ( !(unsigned __int8)sub_158A120(&v14) )
      {
        if ( (unsigned int)a1[4] > 0x40 )
        {
          v6 = *((_QWORD *)a1 + 1);
          if ( v6 )
            j_j___libc_free_0_0(v6);
        }
        v7 = (unsigned int)a1[8] <= 0x40;
        *((_QWORD *)a1 + 1) = v14;
        v8 = v15;
        v15 = 0;
        a1[4] = v8;
        if ( v7 || (v9 = *((_QWORD *)a1 + 3)) == 0 )
        {
          *((_QWORD *)a1 + 3) = v16;
          result = v17;
          a1[8] = v17;
LABEL_7:
          if ( v13 > 0x40 )
          {
            if ( v12 )
              return j_j___libc_free_0_0(v12);
          }
          return result;
        }
        j_j___libc_free_0_0(v9);
        result = v15;
        *((_QWORD *)a1 + 3) = v16;
        a1[8] = v17;
LABEL_21:
        if ( (unsigned int)result > 0x40 && v14 )
          result = j_j___libc_free_0_0(v14);
        goto LABEL_7;
      }
    }
    else if ( !(unsigned __int8)sub_158A120(&v14) )
    {
      v4 = v15;
      *a1 = 3;
      a1[4] = v4;
      *((_QWORD *)a1 + 1) = v14;
      a1[8] = v17;
      result = v16;
      *((_QWORD *)a1 + 3) = v16;
      goto LABEL_7;
    }
    v5 = *a1;
    if ( *a1 != 4 )
    {
      if ( (unsigned int)(v5 - 1) > 1 )
      {
        if ( v5 == 3 )
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
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    result = v15;
    goto LABEL_21;
  }
  if ( (_BYTE)result != 9 )
  {
    *a1 = 1;
    *((_QWORD *)a1 + 1) = a2;
  }
  return result;
}
