// Function: sub_6D70D0
// Address: 0x6d70d0
//
__int64 __fastcall sub_6D70D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 i; // r14
  __int64 v9; // rsi
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 j; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  FILE *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-258h]
  unsigned int v25; // [rsp+1Ch] [rbp-244h] BYREF
  _BYTE v26[16]; // [rsp+20h] [rbp-240h] BYREF
  char v27[160]; // [rsp+30h] [rbp-230h] BYREF
  _QWORD v28[2]; // [rsp+D0h] [rbp-190h] BYREF
  char v29; // [rsp+E0h] [rbp-180h]
  char v30; // [rsp+E1h] [rbp-17Fh]
  _BYTE v31[332]; // [rsp+114h] [rbp-14Ch] BYREF

  v5 = *(__int64 **)(a1 + 16);
  v24 = *v5;
  if ( dword_4F077BC )
  {
    v6 = v5[15];
    for ( i = sub_8D40F0(v6); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    sub_6E1E00(4, v27, 1, 0);
    sub_6E1BE0(v26);
    *(_QWORD *)(qword_4D03C50 + 136LL) = v26;
    sub_6E1C20(a2, 1, v26);
    v9 = 0;
    sub_69ED20((__int64)v28, 0, 0, 1);
    v10 = v28[0];
    if ( v30 == 1
      && !(unsigned int)sub_6ED0A0(v28)
      && (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 176LL) & 8) != 0
      && (v6 == v10 || (v9 = v6, (unsigned int)sub_8D97D0(v10, v6, 32, v14, v15))) )
    {
      v9 = 0;
      if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
        v9 = (unsigned int)sub_8D4C10(v10, dword_4F077C4 != 2);
      v16 = i;
      v17 = sub_6EB190(i, v9, 0, v31, &v25, 1);
      j = v25;
      if ( !v25 )
      {
        if ( v17 )
        {
          v9 = 0;
          v11 = sub_6F5430(v17, 0, i, 0, 1, 1, 0, 0, 1, 0, (__int64)v31);
          *(_QWORD *)(a1 + 32) = sub_6F6F40(v28, 0);
          goto LABEL_8;
        }
        goto LABEL_26;
      }
    }
    else
    {
      if ( (unsigned int)sub_8DD3B0(i) || (v16 = v10, (unsigned int)sub_8DBE70(v10)) )
      {
        if ( dword_4F07590 )
        {
          v9 = 0;
          v22 = sub_6F6F40(v28, 0);
          v12 = 0;
          *(_QWORD *)(a1 + 32) = sub_6E2700(v22);
          v11 = sub_725A70(0);
          goto LABEL_9;
        }
        v11 = sub_725A70(0);
LABEL_8:
        v12 = v11;
        sub_6E2920(v11);
LABEL_9:
        sub_6E2B30(v12, v9);
        return v11;
      }
      if ( !v29 )
        goto LABEL_26;
      v21 = v28[0];
      for ( j = *(unsigned __int8 *)(v28[0] + 140LL); (_BYTE)j == 12; j = *(unsigned __int8 *)(v21 + 140) )
        v21 = *(_QWORD *)(v21 + 160);
      if ( !(_BYTE)j )
      {
LABEL_26:
        v11 = 0;
        goto LABEL_8;
      }
    }
    if ( (unsigned int)sub_6E5430(v16, v9, j, v18, v19) )
    {
      v9 = v24;
      sub_6854E0(0x740u, v24);
    }
    goto LABEL_26;
  }
  if ( (unsigned int)sub_6E5430(a1, a2, &dword_4F077BC, 0, a5) )
  {
    v23 = (FILE *)sub_6E1A20(a2);
    sub_6854C0(0x305u, v23, v24);
  }
  v11 = 0;
  sub_6E1990(a2);
  return v11;
}
