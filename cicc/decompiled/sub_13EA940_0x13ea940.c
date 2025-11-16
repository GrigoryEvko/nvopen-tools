// Function: sub_13EA940
// Address: 0x13ea940
//
__int64 __fastcall sub_13EA940(int *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v4; // eax
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // eax
  int v10; // eax
  __int64 v11; // rdi
  bool v12; // cc
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-38h]
  __int64 v25; // [rsp+40h] [rbp-30h]
  unsigned int v26; // [rsp+48h] [rbp-28h]

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)result == 13 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    v22 = v4;
    if ( v4 > 0x40 )
    {
      sub_16A4FD0(&v21, a2 + 24);
      v18 = *(_DWORD *)(a2 + 32);
      if ( v18 > 0x40 )
      {
        sub_16A4FD0(&v17, a2 + 24);
LABEL_5:
        sub_16A7490(&v17, 1);
        v6 = v18;
        v18 = 0;
        v20 = v6;
        v19 = v17;
        sub_15898E0(&v23, &v19, &v21, v7, v8);
        if ( *a1 == 3 )
        {
          if ( !(unsigned __int8)sub_158A120(&v23) )
          {
            if ( (unsigned int)a1[4] > 0x40 )
            {
              v11 = *((_QWORD *)a1 + 1);
              if ( v11 )
                j_j___libc_free_0_0(v11);
            }
            v12 = (unsigned int)a1[8] <= 0x40;
            *((_QWORD *)a1 + 1) = v23;
            v13 = v24;
            v24 = 0;
            a1[4] = v13;
            if ( v12 || (v14 = *((_QWORD *)a1 + 3)) == 0 )
            {
              *((_QWORD *)a1 + 3) = v25;
              result = v26;
              a1[8] = v26;
LABEL_8:
              if ( v20 > 0x40 && v19 )
                result = j_j___libc_free_0_0(v19);
              if ( v18 > 0x40 && v17 )
                result = j_j___libc_free_0_0(v17);
              if ( v22 > 0x40 )
              {
                if ( v21 )
                  return j_j___libc_free_0_0(v21);
              }
              return result;
            }
            j_j___libc_free_0_0(v14);
            result = v24;
            *((_QWORD *)a1 + 3) = v25;
            a1[8] = v26;
LABEL_28:
            if ( (unsigned int)result > 0x40 && v23 )
              result = j_j___libc_free_0_0(v23);
            goto LABEL_8;
          }
        }
        else if ( !(unsigned __int8)sub_158A120(&v23) )
        {
          v9 = v24;
          *a1 = 3;
          a1[4] = v9;
          *((_QWORD *)a1 + 1) = v23;
          a1[8] = v26;
          result = v25;
          *((_QWORD *)a1 + 3) = v25;
          goto LABEL_8;
        }
        v10 = *a1;
        if ( *a1 != 4 )
        {
          if ( (unsigned int)(v10 - 1) > 1 )
          {
            if ( v10 == 3 )
            {
              if ( (unsigned int)a1[8] > 0x40 )
              {
                v15 = *((_QWORD *)a1 + 3);
                if ( v15 )
                  j_j___libc_free_0_0(v15);
              }
              if ( (unsigned int)a1[4] > 0x40 )
              {
                v16 = *((_QWORD *)a1 + 1);
                if ( v16 )
                  j_j___libc_free_0_0(v16);
              }
            }
          }
          else
          {
            *((_QWORD *)a1 + 1) = 0;
          }
          *a1 = 4;
        }
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        result = v24;
        goto LABEL_28;
      }
    }
    else
    {
      v5 = *(_QWORD *)(a2 + 24);
      v18 = v4;
      v21 = v5;
    }
    v17 = *(_QWORD *)(a2 + 24);
    goto LABEL_5;
  }
  if ( (_BYTE)result != 9 )
  {
    *a1 = 2;
    *((_QWORD *)a1 + 1) = a2;
  }
  return result;
}
