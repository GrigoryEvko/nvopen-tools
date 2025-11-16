// Function: sub_6CEC90
// Address: 0x6cec90
//
__int64 __fastcall sub_6CEC90(__int64 a1, __int64 *a2, _DWORD *a3, _QWORD *a4, __int64 a5)
{
  __int64 *v5; // r14
  char v7; // bl
  __int64 v8; // rax
  bool v9; // bl
  _QWORD *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // r8d
  __int64 v15; // rdx
  __int64 result; // rax
  __int64 v17; // rcx
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // [rsp-10h] [rbp-330h]
  __int64 v24; // [rsp-8h] [rbp-328h]
  _DWORD *v25; // [rsp+8h] [rbp-318h]
  _DWORD *v26; // [rsp+8h] [rbp-318h]
  _DWORD *v27; // [rsp+18h] [rbp-308h]
  int v28; // [rsp+18h] [rbp-308h]
  unsigned int v29; // [rsp+24h] [rbp-2FCh] BYREF
  __int64 v30; // [rsp+28h] [rbp-2F8h] BYREF
  _BYTE v31[352]; // [rsp+30h] [rbp-2F0h] BYREF
  _BYTE v32[76]; // [rsp+190h] [rbp-190h] BYREF
  __int64 v33; // [rsp+1DCh] [rbp-144h]

  v5 = a2;
  v7 = *(_BYTE *)(qword_4D03C50 + 21LL);
  *a3 = 0;
  v8 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 21LL) &= ~0x20u;
  v9 = (v7 & 0x20) != 0;
  if ( a2 )
  {
    v10 = v31;
    sub_6F8AB0((_DWORD)a2, (unsigned int)v31, (unsigned int)v32, 0, (unsigned int)&v30, (unsigned int)&v29, 0);
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
      goto LABEL_3;
    if ( (unsigned int)sub_6E5430(v24, v23, v11, v12, v13) )
      sub_6851C0(0x39u, &v30);
  }
  else
  {
    v10 = (_QWORD *)a1;
    v30 = *(_QWORD *)&dword_4F063F8;
    v17 = dword_4F06650[0];
    v29 = dword_4F06650[0];
    if ( (*(_BYTE *)(v8 + 19) & 0x40) != 0 )
    {
      v27 = a3;
      v18 = sub_6E5430(a1, 0, a3, dword_4F06650[0], a5);
      a3 = v27;
      if ( v18 )
      {
        a1 = 57;
        v25 = v27;
        a2 = &v30;
        sub_6851C0(0x39u, &v30);
        v28 = 1;
        a3 = v25;
      }
      else
      {
        v28 = 1;
      }
    }
    else
    {
      v28 = 0;
    }
    v26 = a3;
    sub_7B8B50(a1, a2, a3, v17);
    if ( word_4F06418[0] == 73 && dword_4D04428 )
    {
      v22 = sub_6BA760(0, 0);
      sub_6E9FE0(v22, v32);
      *v26 = 1;
    }
    else
    {
      sub_69ED20((__int64)v32, 0, 2, 0);
    }
    if ( !v28 )
    {
LABEL_3:
      v14 = dword_4F077C4 != 2
         || !unk_4D044D0
         || (*(_BYTE *)(*v10 + 140LL) & 0xFB) != 8
         || (sub_8D4C10(*v10, 0) & 8) == 0
         || sub_8E3AD0(*v10) == 0;
      sub_6927A0(v10, (__int64)v32, (__int64)&v30, v29, v14, (__int64)a4);
      goto LABEL_6;
    }
  }
  sub_6E6260(a4);
  sub_6E6450(v10);
  sub_6E6450(v32);
  v19 = *((_DWORD *)v10 + 17);
  *((_WORD *)a4 + 36) = *((_WORD *)v10 + 36);
  *((_DWORD *)a4 + 17) = v19;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)((char *)a4 + 68);
  v20 = v33;
  *(_QWORD *)((char *)a4 + 76) = v33;
  *(_QWORD *)&dword_4F061D8 = v20;
  sub_6E3280(a4, &v30);
LABEL_6:
  sub_6E26D0(2, a4);
  v15 = qword_4D03C50;
  *(_BYTE *)(qword_4D03C50 + 21LL) = (32 * v9) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xDF;
  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    result = *(_QWORD *)(v15 + 16) & 0x200000000400LL;
    if ( !result )
    {
      if ( v5 || (result = (__int64)word_4F06418, word_4F06418[0] != 67) )
      {
        result = *(_BYTE *)(*a4 + 140LL) & 0xFB;
        if ( (*(_BYTE *)(*a4 + 140LL) & 0xFB) == 8 )
        {
          result = sub_8D4C10(*a4, 0);
          if ( (result & 2) != 0 )
          {
            result = sub_8D3A70(*a4);
            if ( !(_DWORD)result )
            {
              v21 = 4;
              if ( dword_4F077C4 == 2 )
                v21 = (unsigned int)(unk_4F07778 > 202001) + 4;
              return sub_6E5C80(v21, 3011, (char *)a4 + 68);
            }
          }
        }
      }
    }
  }
  return result;
}
