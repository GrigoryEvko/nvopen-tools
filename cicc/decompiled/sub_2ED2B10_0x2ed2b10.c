// Function: sub_2ED2B10
// Address: 0x2ed2b10
//
__int64 *__fastcall sub_2ED2B10(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 *v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned int v17; // r13d
  __int128 v18; // [rsp+0h] [rbp-60h]
  __int64 *v21; // [rsp+20h] [rbp-40h]
  __int64 *v22; // [rsp+28h] [rbp-38h]

  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 > 0 )
  {
    v22 = a1;
    while ( 1 )
    {
      v8 = (__int64 *)a4[8];
      v9 = &v22[v6 >> 1];
      v10 = *a3;
      v21 = v9;
      v11 = *v9;
      if ( v8 )
      {
        v12 = sub_2E39EA0(v8, *v9);
        v13 = (__int64 *)a4[8];
        *((_QWORD *)&v18 + 1) = v12;
        if ( v13 )
        {
          v14 = sub_2E39EA0(v13, v10);
          v13 = (__int64 *)a4[8];
          v15 = v14;
        }
        else
        {
          v15 = 0;
        }
        *(_QWORD *)&v18 = v15;
        if ( !(unsigned __int8)sub_2EE68A0(*a5, a4[7], v13, 2) && v18 != 0 )
        {
          if ( *((_QWORD *)&v18 + 1) >= (unsigned __int64)v18 )
            goto LABEL_14;
          goto LABEL_9;
        }
      }
      else
      {
        sub_2EE68A0(*a5, a4[7], 0, 2);
      }
      v17 = sub_2E5E7B0(a4[6], v11);
      if ( v17 >= (unsigned int)sub_2E5E7B0(a4[6], v10) )
      {
LABEL_14:
        v6 >>= 1;
        goto LABEL_10;
      }
LABEL_9:
      v6 = v6 - (v6 >> 1) - 1;
      v22 = v21 + 1;
LABEL_10:
      if ( v6 <= 0 )
        return v22;
    }
  }
  return a1;
}
