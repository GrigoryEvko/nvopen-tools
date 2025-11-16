// Function: sub_13E8810
// Address: 0x13e8810
//
__int64 __fastcall sub_13E8810(int *a1, unsigned int *a2)
{
  int v3; // edx
  __int64 result; // rax
  unsigned int v6; // eax
  unsigned int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax

  v3 = *a1;
  if ( *a1 == 3 )
  {
    if ( *a2 == 3 )
      goto LABEL_14;
    if ( (unsigned int)a1[8] > 0x40 )
    {
      v8 = *((_QWORD *)a1 + 3);
      if ( v8 )
        j_j___libc_free_0_0(v8);
    }
    if ( (unsigned int)a1[4] > 0x40 )
    {
      v9 = *((_QWORD *)a1 + 1);
      if ( v9 )
        j_j___libc_free_0_0(v9);
    }
    v3 = *a1;
  }
  result = *a2;
  if ( (unsigned int)(v3 - 1) <= 1 )
  {
    if ( (unsigned int)(result - 1) > 1 )
    {
      *((_QWORD *)a1 + 1) = 0;
      result = *a2;
      if ( (unsigned int)result > 2 )
      {
        if ( (_DWORD)result == 3 )
          goto LABEL_9;
LABEL_6:
        *a1 = result;
        return result;
      }
    }
LABEL_4:
    if ( (_DWORD)result )
    {
      *((_QWORD *)a1 + 1) = *((_QWORD *)a2 + 1);
      result = *a2;
    }
    goto LABEL_6;
  }
  if ( (unsigned int)result <= 2 )
    goto LABEL_4;
  if ( (_DWORD)result != 3 )
    goto LABEL_6;
  if ( v3 == 3 )
  {
LABEL_14:
    if ( (unsigned int)a1[4] <= 0x40 && a2[4] <= 0x40 )
    {
      v13 = *((_QWORD *)a2 + 1);
      *((_QWORD *)a1 + 1) = v13;
      v14 = a2[4];
      a1[4] = v14;
      v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
      if ( (unsigned int)v14 > 0x40 )
      {
        v17 = (unsigned int)((unsigned __int64)(v14 + 63) >> 6) - 1;
        *(_QWORD *)(v13 + 8 * v17) &= v15;
      }
      else
      {
        *((_QWORD *)a1 + 1) = v15 & v13;
      }
    }
    else
    {
      sub_16A51C0(a1 + 2, a2 + 2);
    }
    if ( (unsigned int)a1[8] <= 0x40 && a2[8] <= 0x40 )
    {
      v10 = *((_QWORD *)a2 + 3);
      *((_QWORD *)a1 + 3) = v10;
      v11 = a2[8];
      a1[8] = v11;
      v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
      if ( (unsigned int)v11 > 0x40 )
      {
        v16 = (unsigned int)((unsigned __int64)(v11 + 63) >> 6) - 1;
        *(_QWORD *)(v10 + 8 * v16) &= v12;
      }
      else
      {
        *((_QWORD *)a1 + 3) = v12 & v10;
      }
      result = *a2;
    }
    else
    {
      sub_16A51C0(a1 + 6, a2 + 6);
      result = *a2;
    }
    goto LABEL_6;
  }
LABEL_9:
  v6 = a2[4];
  a1[4] = v6;
  if ( v6 > 0x40 )
    sub_16A4FD0(a1 + 2, a2 + 2);
  else
    *((_QWORD *)a1 + 1) = *((_QWORD *)a2 + 1);
  v7 = a2[8];
  a1[8] = v7;
  if ( v7 > 0x40 )
  {
    sub_16A4FD0(a1 + 6, a2 + 6);
    result = *a2;
    goto LABEL_6;
  }
  *((_QWORD *)a1 + 3) = *((_QWORD *)a2 + 3);
  result = *a2;
  *a1 = result;
  return result;
}
