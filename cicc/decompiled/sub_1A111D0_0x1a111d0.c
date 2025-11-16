// Function: sub_1A111D0
// Address: 0x1a111d0
//
int *__fastcall sub_1A111D0(int *a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  char v6; // al
  unsigned int v7; // eax
  int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // rdi
  bool v12; // cc
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+20h] [rbp-30h]
  unsigned int v22; // [rsp+28h] [rbp-28h]

  v3 = *a2;
  v4 = (v3 >> 1) & 3;
  if ( v4 != 3 )
  {
    if ( (unsigned int)(v4 - 1) > 1 )
    {
      *a1 = 0;
    }
    else
    {
      v5 = v3 & 0xFFFFFFFFFFFFFFF8LL;
      *a1 = 0;
      v6 = *(_BYTE *)(v5 + 16);
      if ( v6 != 9 )
      {
        if ( v6 == 13 )
        {
          v18 = *(_DWORD *)(v5 + 32);
          if ( v18 > 0x40 )
            sub_16A4FD0((__int64)&v17, (const void **)(v5 + 24));
          else
            v17 = *(_QWORD *)(v5 + 24);
          sub_1589870((__int64)&v19, &v17);
          if ( *a1 == 3 )
          {
            if ( !sub_158A120((__int64)&v19) )
            {
              if ( (unsigned int)a1[4] > 0x40 )
              {
                v11 = *((_QWORD *)a1 + 1);
                if ( v11 )
                  j_j___libc_free_0_0(v11);
              }
              v12 = (unsigned int)a1[8] <= 0x40;
              *((_QWORD *)a1 + 1) = v19;
              v13 = v20;
              v20 = 0;
              a1[4] = v13;
              if ( v12 || (v14 = *((_QWORD *)a1 + 3)) == 0 )
              {
                *((_QWORD *)a1 + 3) = v21;
                a1[8] = v22;
LABEL_10:
                if ( v18 > 0x40 )
                {
                  if ( v17 )
                    j_j___libc_free_0_0(v17);
                }
                return a1;
              }
              j_j___libc_free_0_0(v14);
              v10 = v20;
              *((_QWORD *)a1 + 3) = v21;
              a1[8] = v22;
LABEL_25:
              if ( v10 > 0x40 && v19 )
                j_j___libc_free_0_0(v19);
              goto LABEL_10;
            }
          }
          else if ( !sub_158A120((__int64)&v19) )
          {
            v7 = v20;
            *a1 = 3;
            a1[4] = v7;
            *((_QWORD *)a1 + 1) = v19;
            a1[8] = v22;
            *((_QWORD *)a1 + 3) = v21;
            goto LABEL_10;
          }
          v9 = *a1;
          if ( *a1 != 4 )
          {
            if ( (unsigned int)(v9 - 1) > 1 )
            {
              if ( v9 == 3 )
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
          if ( v22 > 0x40 && v21 )
            j_j___libc_free_0_0(v21);
          v10 = v20;
          goto LABEL_25;
        }
        *a1 = 1;
        *((_QWORD *)a1 + 1) = v5;
      }
    }
    return a1;
  }
  *a1 = 4;
  return a1;
}
