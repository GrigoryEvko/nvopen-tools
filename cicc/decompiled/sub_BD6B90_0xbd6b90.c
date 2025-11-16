// Function: sub_BD6B90
// Address: 0xbd6b90
//
void __fastcall sub_BD6B90(unsigned __int8 *a1, unsigned __int8 *a2)
{
  const char *v3; // r15
  __int64 v4; // rax
  const char *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  const char *v9; // rdi
  __int64 v10; // rax
  const char *v11; // [rsp+8h] [rbp-58h] BYREF
  const char *v12[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v13; // [rsp+30h] [rbp-30h]

  v11 = 0;
  if ( (a1[7] & 0x10) == 0 )
  {
    if ( (a2[7] & 0x10) == 0 )
      return;
    goto LABEL_14;
  }
  if ( (unsigned __int8)sub_BD3080(a1, &v11) )
  {
    if ( (a2[7] & 0x10) != 0 )
      goto LABEL_15;
  }
  else
  {
    v3 = v11;
    if ( v11 )
    {
      v4 = sub_BD5C70((__int64)a1);
      sub_BD8AE0(v3, v4);
    }
    sub_BD6840((__int64)a1);
    if ( (a2[7] & 0x10) != 0 )
    {
      if ( v11 )
      {
LABEL_9:
        sub_BD3080(a2, v12);
        v5 = v12[0];
        if ( v11 == v12[0] )
        {
          v10 = sub_BD5C70((__int64)a2);
          sub_BD6500((__int64)a1, v10);
          sub_BD6500((__int64)a2, 0);
          *(_QWORD *)(sub_BD5C70((__int64)a1) + 8) = a1;
        }
        else
        {
          if ( v12[0] )
          {
            v6 = sub_BD5C70((__int64)a2);
            sub_BD8AE0(v5, v6);
          }
          v7 = sub_BD5C70((__int64)a2);
          sub_BD6500((__int64)a1, v7);
          sub_BD6500((__int64)a2, 0);
          v8 = sub_BD5C70((__int64)a1);
          v9 = v11;
          *(_QWORD *)(v8 + 8) = a1;
          if ( v9 )
            sub_BD8920(v9, a1);
        }
        return;
      }
LABEL_14:
      if ( (unsigned __int8)sub_BD3080(a1, &v11) )
      {
LABEL_15:
        v13 = 257;
        sub_BD6B50(a2, v12);
        return;
      }
      goto LABEL_9;
    }
  }
}
