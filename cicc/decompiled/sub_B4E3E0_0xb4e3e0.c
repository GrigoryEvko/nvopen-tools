// Function: sub_B4E3E0
// Address: 0xb4e3e0
//
unsigned __int64 __fastcall sub_B4E3E0(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  char v4; // r13
  int v5; // eax
  int v6; // r12d
  unsigned __int64 result; // rax
  unsigned int i; // r13d
  int v9; // r14d
  unsigned int v10; // r14d
  unsigned __int64 v11; // rdx
  _QWORD *v12; // r13
  unsigned __int8 *v13; // rax
  int v14; // r14d
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  int v17; // r13d
  int v18; // eax
  int *v19; // rdx
  unsigned __int64 v20; // r13

  v2 = *((_QWORD *)a1 + 1);
  v3 = *(unsigned int *)(v2 + 32);
  v4 = *(_BYTE *)(v2 + 8);
  v5 = *a1;
  v6 = v3;
  if ( (_BYTE)v5 == 14 )
  {
    result = *((unsigned int *)a2 + 2);
    if ( result != v3 )
    {
      if ( result <= v3 )
      {
        v20 = v3 - result;
        if ( (unsigned int)v3 > *((_DWORD *)a2 + 3) )
        {
          sub_C8D5F0(a2, a2 + 2, v3, 4);
          result = *((unsigned int *)a2 + 2);
        }
        if ( 4 * v20 )
        {
          memset((void *)(*a2 + 4 * result), 0, 4 * v20);
          result = *((unsigned int *)a2 + 2);
        }
        result += v20;
        *((_DWORD *)a2 + 2) = result;
      }
      else
      {
        *((_DWORD *)a2 + 2) = v3;
      }
    }
  }
  else
  {
    if ( *((_DWORD *)a2 + 3) < (unsigned int)v3 )
    {
      sub_C8D5F0(a2, a2 + 2, v3, 4);
      v5 = *a1;
    }
    if ( v4 != 18 )
    {
      result = (unsigned int)(v5 - 15);
      if ( (unsigned int)result <= 1 )
      {
        if ( v6 )
        {
          for ( i = 0; i != v6; ++i )
          {
            v9 = sub_AC5320((__int64)a1, i);
            result = *((unsigned int *)a2 + 2);
            if ( result + 1 > *((unsigned int *)a2 + 3) )
            {
              sub_C8D5F0(a2, a2 + 2, result + 1, 4);
              result = *((unsigned int *)a2 + 2);
            }
            *(_DWORD *)(*a2 + 4 * result) = v9;
            ++*((_DWORD *)a2 + 2);
          }
        }
        return result;
      }
      if ( !v6 )
        return result;
      v10 = 0;
      while ( 1 )
      {
        LODWORD(v12) = -1;
        v13 = (unsigned __int8 *)sub_AD69F0(a1, v10);
        if ( (unsigned int)*v13 - 12 <= 1 || (v12 = (_QWORD *)*((_QWORD *)v13 + 3), *((_DWORD *)v13 + 8) <= 0x40u) )
        {
          result = *((unsigned int *)a2 + 2);
          v11 = result + 1;
          if ( result + 1 > *((unsigned int *)a2 + 3) )
            goto LABEL_19;
        }
        else
        {
          result = *((unsigned int *)a2 + 2);
          v12 = (_QWORD *)*v12;
          v11 = result + 1;
          if ( result + 1 > *((unsigned int *)a2 + 3) )
          {
LABEL_19:
            sub_C8D5F0(a2, a2 + 2, v11, 4);
            result = *((unsigned int *)a2 + 2);
          }
        }
        ++v10;
        *(_DWORD *)(*a2 + 4 * result) = (_DWORD)v12;
        ++*((_DWORD *)a2 + 2);
        if ( v10 == v6 )
          return result;
      }
    }
    result = (unsigned int)(v5 - 12);
    v14 = -((unsigned int)result < 2);
    if ( v6 )
    {
      v15 = *((unsigned int *)a2 + 2);
      v16 = *((unsigned int *)a2 + 3);
      v17 = 0;
      v18 = *((_DWORD *)a2 + 2);
      if ( v15 >= v16 )
        goto LABEL_27;
LABEL_22:
      v19 = (int *)(*a2 + 4 * v15);
      if ( v19 )
      {
        *v19 = v14;
        v18 = *((_DWORD *)a2 + 2);
      }
      result = (unsigned int)(v18 + 1);
      for ( *((_DWORD *)a2 + 2) = result; v6 != ++v17; ++*((_DWORD *)a2 + 2) )
      {
        v15 = *((unsigned int *)a2 + 2);
        v16 = *((unsigned int *)a2 + 3);
        v18 = *((_DWORD *)a2 + 2);
        if ( v15 < v16 )
          goto LABEL_22;
LABEL_27:
        if ( v16 < v15 + 1 )
        {
          sub_C8D5F0(a2, a2 + 2, v15 + 1, 4);
          v15 = *((unsigned int *)a2 + 2);
        }
        result = *a2;
        *(_DWORD *)(*a2 + 4 * v15) = v14;
      }
    }
  }
  return result;
}
