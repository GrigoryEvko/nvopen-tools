// Function: sub_13FD560
// Address: 0x13fd560
//
_QWORD *__fastcall sub_13FD560(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v21; // [rsp+18h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = sub_13FD000(a2);
  if ( v3 && (v4 = v3, v5 = *(unsigned int *)(v3 + 8), v21 = 0, (unsigned int)v5 > 1) )
  {
    v6 = v5;
    v7 = 1;
    v8 = 0;
    while ( 1 )
    {
      if ( **(_BYTE **)(v4 + 8 * (v7 - v5)) == 5 )
      {
        if ( v8 )
        {
          sub_15C7080(&v22, *(_QWORD *)(v4 + 8 * (v7 - v5)));
          v23[0] = v21;
          if ( v21 )
          {
            sub_1623A60(v23, v21, 2);
            v15 = v23[0];
            *a1 = v23[0];
            if ( v15 )
            {
              sub_1623210(v23, v15, a1);
              v23[0] = 0;
            }
          }
          else
          {
            *a1 = 0;
          }
          v16 = v22;
          a1[1] = v22;
          if ( v16 )
          {
            sub_1623210(&v22, v16, a1 + 1);
            v22 = 0;
          }
          if ( v23[0] )
            sub_161E7C0(v23);
          if ( v22 )
            sub_161E7C0(&v22);
          goto LABEL_16;
        }
        sub_15C7080(v23, *(_QWORD *)(v4 + 8 * (v7 - v5)));
        if ( v21 )
          sub_161E7C0(&v21);
        v8 = v23[0];
        v21 = v23[0];
        if ( v23[0] )
        {
          sub_1623210(v23, v23[0], &v21);
          v8 = v21;
        }
      }
      if ( v6 == ++v7 )
        break;
      v5 = *(unsigned int *)(v4 + 8);
    }
    if ( !v8 )
      goto LABEL_19;
    v23[0] = v8;
    sub_1623A60(v23, v8, 2);
    v9 = v23[0];
    *a1 = v23[0];
    if ( v9 )
      sub_1623210(v23, v9, a1);
    a1[1] = 0;
LABEL_16:
    if ( v21 )
      sub_161E7C0(&v21);
  }
  else
  {
LABEL_19:
    v11 = sub_13FC520(a2);
    if ( v11 && (v12 = *(_QWORD *)(sub_157EBA0(v11) + 48), (v22 = v12) != 0) && (sub_1623A60(&v22, v12, 2), v22) )
    {
      v23[0] = v22;
      sub_1623A60(v23, v22, 2);
      v13 = v23[0];
      *a1 = v23[0];
      if ( v13 )
        sub_1623210(v23, v13, a1);
      v14 = v22;
      a1[1] = 0;
      if ( v14 )
        sub_161E7C0(&v22);
    }
    else
    {
      v17 = **(_QWORD **)(a2 + 32);
      if ( v17 )
      {
        v18 = *(_QWORD *)(sub_157EBA0(v17) + 48);
        v23[0] = v18;
        if ( v18 )
        {
          sub_1623A60(v23, v18, 2);
          v19 = v23[0];
          *a1 = v23[0];
          if ( v19 )
            sub_1623210(v23, v19, a1);
        }
        else
        {
          *a1 = 0;
        }
        a1[1] = 0;
      }
      else
      {
        *a1 = 0;
        a1[1] = 0;
      }
    }
  }
  return a1;
}
