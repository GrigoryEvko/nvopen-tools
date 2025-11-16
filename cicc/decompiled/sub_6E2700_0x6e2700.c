// Function: sub_6E2700
// Address: 0x6e2700
//
__int64 __fastcall sub_6E2700(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _QWORD *v8; // r14
  int v9; // eax
  __int64 v10; // rax
  _QWORD *v11; // rdx
  char v12; // cl
  __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD v15[13]; // [rsp+0h] [rbp-100h] BYREF
  int v16; // [rsp+6Ch] [rbp-94h]

  v6 = a1;
  if ( *(_QWORD *)qword_4D03C50 )
    return v6;
  v8 = *(_QWORD **)(qword_4D03C50 + 48LL);
  if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 4) != 0 )
  {
    sub_76C7C0(v15, a2, a3, a4, a5, a6);
    a2 = v15;
    v15[0] = sub_6E5AE0;
    sub_76CDC0(a1);
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( !dword_4D03A90 )
      return v6;
LABEL_7:
    sub_76C7C0(v15, a2, a3, a4, a5, a6);
    a2 = v15;
    v15[0] = sub_6E0040;
    v15[4] = sub_6E0190;
    v15[5] = sub_6DF600;
    v16 = 1;
    sub_76CDC0(a1);
    v9 = dword_4F077C4;
    if ( dword_4F077C4 != 2 )
      return v6;
    if ( !v8 )
      goto LABEL_9;
LABEL_23:
    if ( !(unsigned int)sub_733920(v8) )
    {
      if ( !*(_BYTE *)(a1 + 24) )
      {
        sub_733CF0(v8, a2);
        v9 = dword_4F077C4;
        goto LABEL_9;
      }
      a2 = v8;
      v6 = sub_7335F0(a1, v8);
    }
    v9 = dword_4F077C4;
LABEL_9:
    if ( v9 != 2 )
      return v6;
    goto LABEL_10;
  }
  a3 = *(_QWORD *)(qword_4D03C50 + 64LL);
  v14 = *(_QWORD *)(qword_4F06BC0 + 24LL);
  if ( v14 != a3 && a3 != *(_QWORD *)(v14 + 32) )
    goto LABEL_7;
  a3 = (unsigned int)dword_4D03A90;
  if ( dword_4D03A90 )
    goto LABEL_7;
  if ( v8 )
    goto LABEL_23;
LABEL_10:
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x42) == 2 && *(_BYTE *)(v6 + 24) == 3 )
  {
    v10 = *(_QWORD *)(v6 + 56);
    if ( (*(_BYTE *)(v10 + 169) & 0x10) == 0 )
    {
      v11 = *(_QWORD **)(v10 + 216);
      if ( v11 )
      {
        v12 = *(_BYTE *)(v10 + 170);
        if ( (v12 & 0x40) == 0 )
        {
          if ( *v11 )
          {
            if ( v12 >= 0 )
            {
              v13 = sub_892240(*(_QWORD *)v10, a2);
              if ( (*(_BYTE *)(*(_QWORD *)(v13 + 16) + 28LL) & 1) == 0 )
                sub_8AA320(v13, 0, 1);
            }
          }
        }
      }
    }
  }
  return v6;
}
