// Function: sub_2ED2C50
// Address: 0x2ed2c50
//
__int64 *__fastcall sub_2ED2C50(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 *v8; // rdi
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned int v16; // r13d
  __int128 v17; // [rsp+0h] [rbp-60h]
  __int64 *v20; // [rsp+20h] [rbp-40h]
  __int64 *v21; // [rsp+28h] [rbp-38h]

  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 > 0 )
  {
    v21 = a1;
    while ( 1 )
    {
      v8 = (__int64 *)a4[8];
      v20 = &v21[v6 >> 1];
      v9 = *v20;
      v10 = *a3;
      if ( v8 )
      {
        v11 = sub_2E39EA0(v8, *a3);
        v12 = (__int64 *)a4[8];
        *((_QWORD *)&v17 + 1) = v11;
        if ( v12 )
        {
          v13 = sub_2E39EA0(v12, v9);
          v12 = (__int64 *)a4[8];
          v14 = v13;
        }
        else
        {
          v14 = 0;
        }
        *(_QWORD *)&v17 = v14;
        if ( !(unsigned __int8)sub_2EE68A0(*a5, a4[7], v12, 2) && v17 != 0 )
        {
          if ( *((_QWORD *)&v17 + 1) < (unsigned __int64)v17 )
            goto LABEL_14;
          goto LABEL_9;
        }
      }
      else
      {
        sub_2EE68A0(*a5, a4[7], 0, 2);
      }
      v16 = sub_2E5E7B0(a4[6], v10);
      if ( v16 < (unsigned int)sub_2E5E7B0(a4[6], v9) )
      {
LABEL_14:
        v6 >>= 1;
        goto LABEL_10;
      }
LABEL_9:
      v6 = v6 - (v6 >> 1) - 1;
      v21 = v20 + 1;
LABEL_10:
      if ( v6 <= 0 )
        return v21;
    }
  }
  return a1;
}
