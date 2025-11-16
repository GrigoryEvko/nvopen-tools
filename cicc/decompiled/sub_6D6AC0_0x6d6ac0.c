// Function: sub_6D6AC0
// Address: 0x6d6ac0
//
_DWORD *__fastcall sub_6D6AC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int *v13; // r14
  __int64 v14; // [rsp+8h] [rbp-238h] BYREF
  int v15; // [rsp+14h] [rbp-22Ch] BYREF
  __int64 v16; // [rsp+18h] [rbp-228h] BYREF
  _BYTE v17[160]; // [rsp+20h] [rbp-220h] BYREF
  _QWORD v18[2]; // [rsp+C0h] [rbp-180h] BYREF
  char v19; // [rsp+D1h] [rbp-16Fh]
  _BYTE v20[8]; // [rsp+104h] [rbp-13Ch] BYREF
  __int64 v21; // [rsp+10Ch] [rbp-134h]
  _BYTE v22[128]; // [rsp+150h] [rbp-F0h] BYREF
  __int64 v23; // [rsp+1D0h] [rbp-70h]

  v14 = a1;
  sub_6E2250(v17, &v16, 3, 1, a2, 0);
  sub_69ED20((__int64)v18, 0, 0, 1);
  if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D3410(v14) || !(unsigned int)sub_8D3410(v18[0]) )
  {
LABEL_3:
    sub_843C40((unsigned int)v18, v14, 0, 0, 1, 1, 144);
    sub_6F4950(v18, a3, v4, v5, v6, v7);
    if ( (unsigned int)sub_6E4B40(a3, v20) )
      sub_72C970(a3);
    goto LABEL_5;
  }
  if ( v19 == 1 && !(unsigned int)sub_6ED0A0(v18) )
  {
    if ( sub_694910(v18) )
    {
      v15 = 0;
      if ( dword_4F077C0 )
      {
        if ( !(_DWORD)qword_4F077B4 )
        {
          v13 = &v15;
          if ( !qword_4F077A8 )
            v13 = 0;
          goto LABEL_18;
        }
      }
      else
      {
        v13 = 0;
        if ( !(_DWORD)qword_4F077B4 )
        {
LABEL_18:
          if ( (unsigned int)sub_8D3770(v14) && (unsigned int)sub_631DE0(&v14, (__int64)v22, v13) )
          {
            if ( v15 )
            {
              sub_6E5D70(5, 144, v20, v18[0], v14);
              v18[0] = v23;
            }
            sub_72A510(v22, a3);
          }
          else
          {
            sub_6E5ED0(144, v20, v18[0], v14);
            sub_72C970(a3);
          }
          goto LABEL_5;
        }
      }
      v13 = 0;
      if ( dword_4F077C4 != 2 && qword_4F077A0 )
        v13 = &v15;
      goto LABEL_18;
    }
    goto LABEL_3;
  }
  if ( v18[0] != v14 && !(unsigned int)sub_8DED30(v18[0], v14, 1) )
  {
    sub_6E5ED0(144, v20, v18[0], v14);
    sub_6E6840(v18);
  }
  sub_6F4950(v18, a3, v9, v10, v11, v12);
LABEL_5:
  sub_6E2AC0(a3);
  sub_6E2C70(v16, 1, a2, 0);
  *(_QWORD *)&dword_4F061D8 = v21;
  return &dword_4F061D8;
}
