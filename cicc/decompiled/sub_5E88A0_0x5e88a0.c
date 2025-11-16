// Function: sub_5E88A0
// Address: 0x5e88a0
//
__int64 __fastcall sub_5E88A0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rbx
  int v6; // r12d
  __int64 v7; // r15
  __int64 v8; // rbx
  bool v9; // r12
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // r10
  char v13; // al
  __int64 v14; // r15
  __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // r10
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned int v22; // [rsp+28h] [rbp-38h] BYREF
  _BYTE v23[52]; // [rsp+2Ch] [rbp-34h] BYREF

  result = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v5 = *(_QWORD *)result;
  if ( *(_QWORD *)result )
  {
    v6 = 0;
    result = 0;
    while ( 1 )
    {
      while ( *(_BYTE *)(v5 + 80) != 8 )
      {
LABEL_3:
        v5 = *(_QWORD *)(v5 + 16);
        if ( !v5 )
          goto LABEL_7;
      }
      v7 = *(_QWORD *)(v5 + 88);
      if ( (*(_BYTE *)(v7 + 145) & 0x20) == 0 )
        break;
LABEL_6:
      v5 = *(_QWORD *)(v5 + 16);
      v6 = 1;
      result = 1;
      if ( !v5 )
      {
LABEL_7:
        if ( v6 || *(_BYTE *)(a1 + 140) != 11 )
          goto LABEL_8;
        if ( (_DWORD)result )
          goto LABEL_16;
        if ( a2 )
          goto LABEL_9;
        return result;
      }
    }
    v10 = *(_QWORD *)(v7 + 120);
    if ( (unsigned int)sub_8D2FB0(v10) )
      goto LABEL_34;
    if ( (unsigned int)sub_8D3410(v10) )
    {
      v10 = sub_8D40F0(v10);
      v11 = *(_BYTE *)(v10 + 140);
      if ( v11 == 12 )
        goto LABEL_21;
    }
    else
    {
      v11 = *(_BYTE *)(v10 + 140);
      if ( v11 == 12 )
      {
LABEL_21:
        v12 = v10;
        do
        {
          v12 = *(_QWORD *)(v12 + 160);
          v13 = *(_BYTE *)(v12 + 140);
        }
        while ( v13 == 12 );
        if ( (unsigned __int8)(v13 - 9) <= 2u )
          goto LABEL_24;
        goto LABEL_40;
      }
    }
    if ( (unsigned __int8)(v11 - 9) <= 2u )
    {
      v12 = v10;
LABEL_24:
      v14 = *(_QWORD *)(*(_QWORD *)v12 + 96LL);
      v15 = *(_QWORD *)(v14 + 16);
      if ( v15 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v15 + 88) + 206LL) & 0x10) != 0
          || (v19 = v12, v16 = sub_884000(v15, 1), v17 = v19, !v16) )
        {
LABEL_34:
          result = a3;
          *(_DWORD *)(a3 + 8) = 1;
          if ( v6 || *(_BYTE *)(a1 + 140) != 11 )
            return result;
          goto LABEL_16;
        }
      }
      else
      {
        v21 = v12;
        sub_87CAB0(v12, (unsigned int)&dword_4F063F8, v12, 1, 1, 1, 0, (__int64)&v22, (__int64)v23);
        v17 = v21;
        if ( v22 )
          goto LABEL_34;
      }
      if ( (*(_BYTE *)(v14 + 176) & 4) != 0 || dword_4F077BC && !dword_4F077B4 )
        goto LABEL_6;
      if ( (*(_BYTE *)(v10 + 140) & 0xFB) != 8 )
        goto LABEL_6;
      v20 = v17;
      if ( (sub_8D4C10(v10, unk_4F077C4 != 2) & 1) == 0 )
        goto LABEL_6;
      if ( dword_4F077B4 || !(unsigned int)sub_8D5A50(v20) )
        goto LABEL_34;
LABEL_42:
      result = 1;
      goto LABEL_3;
    }
    if ( (v11 & 0xFB) != 8 )
      goto LABEL_6;
LABEL_40:
    if ( (sub_8D4C10(v10, unk_4F077C4 != 2) & 1) == 0 )
      goto LABEL_6;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 32LL) + 140LL) != 11 )
      goto LABEL_34;
    goto LABEL_42;
  }
LABEL_8:
  if ( a2 )
  {
LABEL_9:
    result = a3;
    if ( !*(_DWORD *)(a3 + 8) )
    {
      result = *(_QWORD *)(a1 + 168);
      v8 = *(_QWORD *)(result + 8);
      v9 = (*(_BYTE *)(a1 + 176) & 0x20) != 0;
      if ( v8 )
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v8 + 96) & 2) == 0 || !v9 && (result = sub_8E35E0(v8, a1), !(_DWORD)result) )
          {
            sub_87CAB0(*(_QWORD *)(v8 + 40), (unsigned int)&dword_4F063F8, a1, 1, 1, 1, 0, (__int64)&v22, (__int64)v23);
            result = v22;
            if ( v22 )
              break;
          }
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            return result;
        }
LABEL_16:
        result = a3;
        *(_DWORD *)(a3 + 8) = 1;
      }
    }
  }
  return result;
}
