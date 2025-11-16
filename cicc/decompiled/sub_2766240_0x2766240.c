// Function: sub_2766240
// Address: 0x2766240
//
__int64 __fastcall sub_2766240(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v5; // r14
  __int64 v6; // rdi
  __int64 *v7; // r12
  bool v8; // al
  __int64 v9; // rsi
  bool v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 *v15; // r12
  __int64 *v16; // r15
  __int64 *v17; // r13
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdi
  bool v25; // al
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 v33; // rsi
  bool v34; // al
  __int64 v35; // rsi
  bool v36; // al
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 *v39; // [rsp+0h] [rbp-40h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v40 = a3;
  if ( (char *)a2 - (char *)a1 > 256 )
  {
    v5 = a2;
    if ( a3 )
    {
      v39 = a1 + 2;
      while ( 1 )
      {
        v6 = a1[2];
        --v40;
        v7 = &a1[2 * (result >> 5)];
        if ( v6 == *v7 )
          v8 = sub_B445A0(a1[3], v7[1]);
        else
          v8 = sub_B445A0(v6, *v7);
        v9 = *(v5 - 2);
        if ( v8 )
        {
          if ( *v7 == v9 )
            v10 = sub_B445A0(v7[1], *(v5 - 1));
          else
            v10 = sub_B445A0(*v7, v9);
          if ( !v10 )
          {
            v32 = a1[2];
            v33 = *(v5 - 2);
            if ( v32 == v33 )
              v34 = sub_B445A0(a1[3], *(v5 - 1));
            else
              v34 = sub_B445A0(v32, v33);
            v13 = *a1;
            if ( v34 )
            {
              *a1 = *(v5 - 2);
              *(v5 - 2) = v13;
LABEL_46:
              v38 = a1[1];
              a1[1] = *(v5 - 1);
              *(v5 - 1) = v38;
              v13 = a1[2];
              v14 = *a1;
              goto LABEL_12;
            }
            goto LABEL_31;
          }
          v11 = *a1;
          *a1 = *v7;
          *v7 = v11;
        }
        else
        {
          v24 = a1[2];
          if ( v9 == v24 )
            v25 = sub_B445A0(a1[3], *(v5 - 1));
          else
            v25 = sub_B445A0(v24, v9);
          if ( v25 )
          {
            v13 = *a1;
LABEL_31:
            v14 = a1[2];
            v26 = a1[1];
            a1[2] = v13;
            v27 = a1[3];
            *a1 = v14;
            a1[1] = v27;
            a1[3] = v26;
            goto LABEL_12;
          }
          v35 = *(v5 - 2);
          if ( *v7 == v35 )
            v36 = sub_B445A0(v7[1], *(v5 - 1));
          else
            v36 = sub_B445A0(*v7, v35);
          v37 = *a1;
          if ( v36 )
          {
            *a1 = *(v5 - 2);
            *(v5 - 2) = v37;
            goto LABEL_46;
          }
          *a1 = *v7;
          *v7 = v37;
        }
        v12 = a1[1];
        a1[1] = v7[1];
        v7[1] = v12;
        v13 = a1[2];
        v14 = *a1;
LABEL_12:
        v15 = v39;
        v16 = v5;
        while ( 1 )
        {
          v17 = v15;
          if ( !(v14 == v13 ? sub_B445A0(v15[1], a1[1]) : sub_B445A0(v13, v14)) )
            break;
LABEL_13:
          v14 = *a1;
          v13 = v15[2];
          v15 += 2;
        }
        v19 = *a1;
        v20 = *(v16 - 2);
        v16 -= 2;
        if ( *a1 != v20 )
        {
LABEL_18:
          if ( !sub_B445A0(v19, v20) )
            goto LABEL_21;
          goto LABEL_19;
        }
        while ( sub_B445A0(a1[1], v16[1]) )
        {
LABEL_19:
          v19 = *a1;
          v20 = *(v16 - 2);
          v16 -= 2;
          if ( *a1 != v20 )
            goto LABEL_18;
        }
LABEL_21:
        if ( v15 < v16 )
        {
          v21 = *v15;
          *v15 = *v16;
          v22 = v16[1];
          *v16 = v21;
          v23 = v15[1];
          v15[1] = v22;
          v16[1] = v23;
          goto LABEL_13;
        }
        sub_2766240(v15, v5, v40);
        result = (char *)v15 - (char *)a1;
        if ( (char *)v15 - (char *)a1 <= 256 )
          return result;
        if ( !v40 )
          goto LABEL_34;
        v5 = v15;
      }
    }
    v17 = a2;
LABEL_34:
    v28 = result >> 4;
    v29 = ((result >> 4) - 2) >> 1;
    sub_2765BC0((__int64)a1, v29, result >> 4, a1[2 * v29], a1[2 * v29 + 1]);
    do
    {
      --v29;
      sub_2765BC0((__int64)a1, v29, v28, a1[2 * v29], a1[2 * v29 + 1]);
    }
    while ( v29 );
    do
    {
      v17 -= 2;
      v30 = *v17;
      v31 = v17[1];
      *v17 = *a1;
      v17[1] = a1[1];
      result = sub_2765BC0((__int64)a1, 0, ((char *)v17 - (char *)a1) >> 4, v30, v31);
    }
    while ( (char *)v17 - (char *)a1 > 16 );
  }
  return result;
}
