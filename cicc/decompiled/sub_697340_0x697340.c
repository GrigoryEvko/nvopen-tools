// Function: sub_697340
// Address: 0x697340
//
__int64 __fastcall sub_697340(_QWORD *a1, __int64 a2, unsigned int a3, int a4, int a5, int a6, __int64 a7)
{
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // rcx
  char v20; // dl
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // r8
  char v25; // al
  __int64 result; // rax
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // [rsp-10h] [rbp-60h]
  __int64 v33; // [rsp-8h] [rbp-58h]
  int v36; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v37[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v36 = 0;
  if ( word_4D04898 )
  {
    if ( (unsigned int)sub_8D3A70(*a1) && (unsigned int)sub_8D4160(*a1) )
    {
      v27 = 526336;
      if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
        v27 = (a4 | a6) == 0 ? 526336 : 0x80000;
      sub_845C60(a1, a2, a3, v27, &v36);
    }
    if ( v36 )
      goto LABEL_40;
  }
  v37[0] = 0;
  sub_6F69D0(a1, 0);
  if ( word_4D04898 && *((_BYTE *)a1 + 16) == 1 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v11 + 6) & 6) == 0 && *(_BYTE *)(v11 + 4) != 12 )
        goto LABEL_7;
    }
    if ( !(unsigned int)sub_696840((__int64)a1) )
    {
      if ( (*(_WORD *)(qword_4D03C50 + 18LL) & 0x240) != 0 )
      {
LABEL_7:
        sub_6F4D20(a1, 1, 1);
        goto LABEL_8;
      }
      if ( (unsigned int)sub_6F4D20(a1, 0, 1) )
        goto LABEL_8;
    }
    return sub_6F4910(a1, a7, a2, v28, v29, v30);
  }
LABEL_8:
  if ( a4 && *((_BYTE *)a1 + 16) == 2 && *((_BYTE *)a1 + 317) == 1 && (int)sub_6210B0((__int64)(a1 + 18), 0) < 0 )
    sub_6E68E0(94, a1);
  if ( !a5 )
  {
    if ( a2 )
      goto LABEL_47;
LABEL_53:
    v14 = a3;
    if ( !(unsigned int)sub_831460(*a1, a3) )
    {
      v25 = *((_BYTE *)a1 + 16);
      if ( !v25 )
        goto LABEL_42;
      v13 = *a1;
      v16 = *(unsigned __int8 *)(*a1 + 140LL);
      if ( (_BYTE)v16 == 12 )
      {
        v15 = *a1;
        do
        {
          v15 = *(_QWORD *)(v15 + 160);
          v16 = *(unsigned __int8 *)(v15 + 140);
        }
        while ( (_BYTE)v16 == 12 );
      }
      if ( !(_BYTE)v16 )
        goto LABEL_41;
      if ( !(unsigned int)sub_8D3D40(v13) )
        goto LABEL_16;
    }
    goto LABEL_40;
  }
  if ( !a2 )
    goto LABEL_53;
  if ( !word_4D04898 )
    goto LABEL_48;
  v12 = *((_BYTE *)a1 + 16);
  if ( v12 != 2 )
    goto LABEL_15;
  if ( (unsigned int)sub_8D2780(*a1) && (unsigned int)sub_8D2780(a2) )
  {
LABEL_40:
    v25 = *((_BYTE *)a1 + 16);
    goto LABEL_41;
  }
LABEL_47:
  if ( !word_4D04898 )
    goto LABEL_48;
  v12 = *((_BYTE *)a1 + 16);
LABEL_15:
  v13 = *a1;
  v14 = v12 == 2;
  if ( (unsigned int)sub_8DD4B0(*a1, v14, a1 + 18, a2, v37) )
    goto LABEL_48;
LABEL_16:
  v18 = v37[0];
  if ( !v37[0] )
    v37[0] = 2373;
  if ( (unsigned int)sub_6E5430(v13, v14, v15, v16, v18, v17) && *((_BYTE *)a1 + 16) )
  {
    v19 = *a1;
    v20 = *(_BYTE *)(*a1 + 140LL);
    if ( v20 == 12 )
    {
      v21 = *a1;
      do
      {
        v21 = *(_QWORD *)(v21 + 160);
        v20 = *(_BYTE *)(v21 + 140);
      }
      while ( v20 == 12 );
    }
    if ( v20 )
    {
      if ( a2 )
      {
        v22 = *(_BYTE *)(a2 + 140);
        if ( v22 == 12 )
        {
          v23 = a2;
          do
          {
            v23 = *(_QWORD *)(v23 + 160);
            v22 = *(_BYTE *)(v23 + 140);
          }
          while ( v22 == 12 );
        }
        if ( v22 )
        {
          if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
            goto LABEL_35;
          if ( qword_4F077A8 > 0x1869Fu || v37[0] != 2362 )
          {
            if ( !a5 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
              goto LABEL_35;
            if ( !(unsigned int)sub_8D28B0(*a1) )
            {
              v19 = *a1;
LABEL_35:
              sub_685750(8u, v37[0], (_DWORD *)a1 + 17, v19, a2);
              goto LABEL_36;
            }
            sub_685330(0xCEFu, (_DWORD *)a1 + 17, *a1);
            sub_6FC3F0(a2, a1, 1);
            v25 = *((_BYTE *)a1 + 16);
LABEL_41:
            if ( v25 != 1 )
              goto LABEL_42;
            goto LABEL_37;
          }
LABEL_48:
          sub_843C40((_DWORD)a1, a2, 0, 0, 1, 0, 2373);
          v15 = v32;
          v25 = *((_BYTE *)a1 + 16);
          v16 = v33;
          goto LABEL_41;
        }
      }
      else
      {
        v31 = sub_82E970(a3);
        sub_686490(0x96Cu, (_DWORD *)a1 + 17, *a1, v31);
      }
    }
  }
LABEL_36:
  sub_6E6840(a1);
  if ( *((_BYTE *)a1 + 16) == 1 )
LABEL_37:
    sub_6E6B60(a1, 0);
LABEL_42:
  result = sub_6F4950(a1, a7, v15, v16, v24, v17);
  if ( a5 )
  {
    result = *(unsigned __int8 *)(a7 + 173);
    if ( (_BYTE)result != 12 && (unsigned __int8)result > 1u )
    {
      sub_6851C0(0xB37u, (_DWORD *)a1 + 17);
      return sub_72C970(a7);
    }
  }
  return result;
}
