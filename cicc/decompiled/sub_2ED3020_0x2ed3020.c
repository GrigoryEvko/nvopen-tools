// Function: sub_2ED3020
// Address: 0x2ed3020
//
char *__fastcall sub_2ED3020(
        __int64 *src,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        _QWORD *a8)
{
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 *v17; // rdi
  __int64 v18; // r15
  __int64 v19; // r14
  unsigned int v20; // r14d
  char *v21; // r8
  __int128 v23; // [rsp+0h] [rbp-60h]
  __int64 *v25; // [rsp+28h] [rbp-38h]

  v8 = a3;
  v9 = src;
  v10 = a5;
  if ( src != a2 && a3 != a4 )
  {
    v25 = a5;
    while ( 1 )
    {
      v17 = (__int64 *)a7[8];
      v18 = *v9;
      v19 = *v8;
      if ( v17 )
      {
        v11 = sub_2E39EA0(v17, *v8);
        v12 = (__int64 *)a7[8];
        *((_QWORD *)&v23 + 1) = v11;
        if ( v12 )
        {
          v13 = sub_2E39EA0(v12, v18);
          v12 = (__int64 *)a7[8];
          v14 = v13;
        }
        else
        {
          v14 = 0;
        }
        *(_QWORD *)&v23 = v14;
        if ( !(unsigned __int8)sub_2EE68A0(*a8, a7[7], v12, 2) && v23 != 0 )
        {
          if ( *((_QWORD *)&v23 + 1) < (unsigned __int64)v23 )
            goto LABEL_9;
          goto LABEL_15;
        }
      }
      else
      {
        sub_2EE68A0(*a8, a7[7], 0, 2);
      }
      v20 = sub_2E5E7B0(a7[6], v19);
      if ( v20 < (unsigned int)sub_2E5E7B0(a7[6], v18) )
      {
LABEL_9:
        v15 = *v8++;
        goto LABEL_10;
      }
LABEL_15:
      v15 = *v9++;
LABEL_10:
      *v25 = v15;
      v16 = ++v25;
      if ( v9 == a2 )
      {
        v10 = v25;
        break;
      }
      if ( v8 == a4 )
      {
        v10 = v16;
        break;
      }
    }
  }
  if ( a2 != v9 )
    memmove(v10, v9, (char *)a2 - (char *)v9);
  v21 = (char *)v10 + (char *)a2 - (char *)v9;
  if ( a4 != v8 )
    v21 = (char *)memmove(v21, v8, (char *)a4 - (char *)v8);
  return &v21[(char *)a4 - (char *)v8];
}
