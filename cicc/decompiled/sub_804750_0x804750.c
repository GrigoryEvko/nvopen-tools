// Function: sub_804750
// Address: 0x804750
//
void __fastcall sub_804750(__int64 a1, __m128i *a2)
{
  __m128i *v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rax
  int v7; // eax
  char *v8; // r8
  int v9; // ecx
  char v10; // al
  unsigned __int8 v11; // di
  unsigned __int8 v12; // al
  __m128i *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rbx
  _BOOL4 v17; // [rsp+1Ch] [rbp-A4h] BYREF
  _BYTE v18[32]; // [rsp+20h] [rbp-A0h] BYREF
  __m128i v19[8]; // [rsp+40h] [rbp-80h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 72);
  v5 = *(_QWORD *)(v4 + 8);
  sub_802F60(v4, 0, 0);
  if ( *(_QWORD *)(v4 + 16)
    || *(_QWORD *)(v4 + 40)
    || *(_BYTE *)(v5 + 136) <= 2u
    || qword_4F04C50 && *(_QWORD *)(qword_4F04C50 + 72LL) == v5
    || (*(_BYTE *)(v5 + 173) & 4) != 0
    || (*(_BYTE *)(v4 + 51) & 2) != 0 )
  {
    if ( *(_BYTE *)(v4 + 48) <= 8u )
      goto LABEL_3;
    goto LABEL_40;
  }
  v12 = *(_BYTE *)(v4 + 48);
  if ( v12 > 3u )
  {
    if ( (unsigned __int8)(v12 - 4) <= 4u )
    {
LABEL_3:
      if ( !a2 )
      {
        v3 = (__m128i *)v18;
        sub_7E1720(a1, (__int64)v18);
      }
      if ( qword_4F04C50 && *(_QWORD *)(qword_4F04C50 + 72LL) == v5 )
      {
        sub_7F90D0(qword_4D03F58, (__int64)v19);
        *(_QWORD *)(v4 + 8) = 0;
      }
      else
      {
        sub_7F9080(v5, (__int64)v19);
      }
      if ( qword_4F04C50
        && (v6 = *(_QWORD *)(qword_4F04C50 + 32LL)) != 0
        && (*(_BYTE *)(v6 + 198) & 0x10) != 0
        && (*(_BYTE *)(v5 + 156) & 1) != 0 )
      {
        v7 = sub_7FA8C0(v4);
        v8 = "__constant__";
        v9 = v7;
        v10 = *(_BYTE *)(v5 + 156);
        if ( (v10 & 4) == 0 )
        {
          v8 = "__managed__";
          if ( (*(_BYTE *)(v5 + 157) & 1) == 0 )
          {
            v8 = "__shared__";
            if ( (v10 & 2) == 0 )
            {
              v8 = (char *)byte_3F871B3;
              if ( (v10 & 1) != 0 )
                v8 = "__device__";
            }
          }
        }
        v11 = 4;
        if ( !v9 )
          v11 = (v10 & 2) == 0 ? 8 : 5;
        sub_6849F0(v11, 0xDB7u, (_DWORD *)(*(_QWORD *)(v4 + 8) + 64LL), (__int64)v8);
        sub_7259F0(v4, 0);
        v17 = 0;
      }
      else
      {
        sub_7FEC50(v4, v19, 0, 0, 1, 0, v3, &v17, 0);
        if ( v17 )
          return;
      }
      sub_7F8B60(a1);
      return;
    }
LABEL_40:
    sub_721090();
  }
  switch ( v12 )
  {
    case 1u:
      goto LABEL_3;
    case 2u:
      sub_7EB190(*(_QWORD *)(v4 + 56), 0);
      v15 = *(_QWORD *)(v4 + 56);
      if ( *(_BYTE *)(v15 + 173) != 6 || *(_BYTE *)(v15 + 176) || dword_4F077C4 != 2 || dword_4F06968 )
      {
        sub_7F5C00(v15);
      }
      else if ( (unsigned int)sub_7E1F90(*(_QWORD *)(v15 + 128))
             || !(unsigned int)sub_7E6740(*(_QWORD *)(*(_QWORD *)(v4 + 56) + 128LL)) )
      {
        sub_7F5C00(*(_QWORD *)(v4 + 56));
      }
      else
      {
        v16 = *(_QWORD *)(v4 + 56);
        *(_QWORD *)(v16 + 128) = sub_7E6760(*(const __m128i **)(v16 + 128), *(_QWORD *)(v5 + 120));
        *(_BYTE *)(*(_QWORD *)(v4 + 56) + 168LL) |= 8u;
      }
      break;
    case 3u:
      sub_7F2600(*(_QWORD *)(v4 + 56), 0);
      if ( dword_4F077C4 == 2
        && !dword_4F06968
        && !(unsigned int)sub_7E1F90(**(_QWORD **)(v4 + 56))
        && (unsigned int)sub_7E6740(**(_QWORD **)(v4 + 56)) )
      {
        v13 = sub_7E6760(**(const __m128i ***)(v4 + 56), *(_QWORD *)(v5 + 120));
        v14 = sub_73E110(*(_QWORD *)(v4 + 56), (__int64)v13);
        *(_QWORD *)(v4 + 56) = v14;
        sub_7E67B0(v14);
      }
      break;
    default:
      sub_7F8B60(a1);
      *(_BYTE *)(v5 + 177) = 0;
      break;
  }
}
