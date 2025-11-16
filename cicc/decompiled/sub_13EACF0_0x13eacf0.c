// Function: sub_13EACF0
// Address: 0x13eacf0
//
__int64 __fastcall sub_13EACF0(int *a1, unsigned int *a2)
{
  __int64 result; // rax
  int v4; // edx
  unsigned int v5; // ebx
  unsigned int v6; // r15d
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v13; // [rsp-70h] [rbp-70h]
  __int64 v14; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v15; // [rsp-60h] [rbp-60h]
  __int64 v16; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v17; // [rsp-50h] [rbp-50h]
  __int64 v18; // [rsp-48h] [rbp-48h]
  unsigned int v19; // [rsp-40h] [rbp-40h]

  result = *a2;
  if ( !(_DWORD)result )
    return result;
  v4 = *a1;
  if ( *a1 == 4 )
    return result;
  if ( (_DWORD)result == 4 )
  {
    result = (unsigned int)(v4 - 1);
    if ( (unsigned int)result > 1 )
    {
LABEL_34:
      if ( v4 == 3 )
      {
        if ( (unsigned int)a1[8] > 0x40 )
        {
          v8 = *((_QWORD *)a1 + 3);
          if ( v8 )
            result = j_j___libc_free_0_0(v8);
        }
        if ( (unsigned int)a1[4] > 0x40 )
        {
          v9 = *((_QWORD *)a1 + 1);
          if ( v9 )
            result = j_j___libc_free_0_0(v9);
        }
      }
      goto LABEL_29;
    }
    goto LABEL_28;
  }
  switch ( v4 )
  {
    case 0:
      return sub_13E8810(a1, a2);
    case 1:
      if ( (_DWORD)result == 1 )
      {
LABEL_27:
        result = *((_QWORD *)a2 + 1);
        if ( *((_QWORD *)a1 + 1) == result )
          return result;
      }
LABEL_28:
      *((_QWORD *)a1 + 1) = 0;
LABEL_29:
      *a1 = 4;
      return result;
    case 2:
      if ( (_DWORD)result == 2 )
        goto LABEL_27;
      goto LABEL_28;
  }
  if ( (_DWORD)result != 3 )
    goto LABEL_34;
  sub_158C3A0(&v12, a1 + 2, a2 + 2);
  if ( (unsigned __int8)sub_158A0B0(&v12) )
  {
    result = (unsigned int)*a1;
    if ( (_DWORD)result != 4 )
    {
      if ( (unsigned int)(result - 1) > 1 )
      {
        if ( (_DWORD)result == 3 )
        {
          if ( (unsigned int)a1[8] > 0x40 )
          {
            v10 = *((_QWORD *)a1 + 3);
            if ( v10 )
              result = j_j___libc_free_0_0(v10);
          }
          if ( (unsigned int)a1[4] > 0x40 )
          {
            v11 = *((_QWORD *)a1 + 1);
            if ( v11 )
              result = j_j___libc_free_0_0(v11);
          }
        }
      }
      else
      {
        *((_QWORD *)a1 + 1) = 0;
      }
      *a1 = 4;
    }
LABEL_19:
    if ( v15 <= 0x40 )
    {
LABEL_20:
      v5 = v13;
      goto LABEL_21;
    }
LABEL_48:
    if ( v14 )
      result = j_j___libc_free_0_0(v14);
    goto LABEL_20;
  }
  v5 = v13;
  v6 = v15;
  if ( v13 <= 0x40 )
  {
    result = v12;
    if ( v12 != *((_QWORD *)a1 + 1) )
      goto LABEL_12;
  }
  else
  {
    result = sub_16A5220(&v12, a1 + 2);
    if ( !(_BYTE)result )
    {
LABEL_11:
      result = v12;
LABEL_12:
      v7 = v14;
      goto LABEL_13;
    }
  }
  if ( v6 > 0x40 )
  {
    result = sub_16A5220(&v14, a1 + 6);
    if ( (_BYTE)result )
      goto LABEL_48;
    goto LABEL_11;
  }
  v7 = v14;
  if ( v14 != *((_QWORD *)a1 + 3) )
  {
    result = v12;
LABEL_13:
    v17 = v5;
    v16 = result;
    v13 = 0;
    v19 = v6;
    v18 = v7;
    v15 = 0;
    result = sub_13EABD0((unsigned int *)a1, (__int64)&v16);
    if ( v19 > 0x40 && v18 )
      result = j_j___libc_free_0_0(v18);
    if ( v17 > 0x40 && v16 )
      result = j_j___libc_free_0_0(v16);
    goto LABEL_19;
  }
LABEL_21:
  if ( v5 > 0x40 )
  {
    if ( v12 )
      return j_j___libc_free_0_0(v12);
  }
  return result;
}
