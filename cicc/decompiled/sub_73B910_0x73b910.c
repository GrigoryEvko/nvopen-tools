// Function: sub_73B910
// Address: 0x73b910
//
void __fastcall sub_73B910(__int64 a1)
{
  __int64 v2; // rax
  char v3; // al
  char v4; // bl
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  if ( (*(_BYTE *)(a1 - 8) & 1) == 0 )
    return;
  v2 = *(_QWORD *)(a1 + 144);
  if ( v2 && (*(_BYTE *)(v2 - 8) & 1) == 0 )
    *(_QWORD *)(a1 + 144) = 0;
  v3 = *(_BYTE *)(a1 + 173);
  if ( v3 == 12 )
  {
    v4 = *(_BYTE *)(a1 + 176);
    if ( (unsigned __int8)(v4 - 5) <= 5u )
    {
      if ( v4 != 1 )
      {
        v5 = *(_QWORD *)(a1 + 192);
        v6 = (__int64 *)(a1 + 192);
        goto LABEL_12;
      }
    }
    else if ( v4 != 1 )
    {
      return;
    }
    v5 = *(_QWORD *)(a1 + 184);
    v6 = (__int64 *)(a1 + 184);
LABEL_12:
    if ( v5 && (*(_BYTE *)(v5 - 8) & 1) == 0 )
    {
      v7 = *(_QWORD *)(a1 + 48);
      if ( v7 && (v8 = sub_72B800(v7)) != 0 || (v8 = sub_867B10()) != 0 )
      {
        sub_72D910(*v6, (v4 == 1) + 2, a1, v8);
        *v6 = 0;
      }
      else
      {
        *v6 = (__int64)sub_73B8B0((const __m128i *)*v6, (*(unsigned __int8 *)(qword_4D03C50 + 19LL) << 13) & 0x4000);
      }
    }
    return;
  }
  if ( v3 == 15 )
  {
    v9 = *(_QWORD *)(a1 + 184);
    if ( v9 )
    {
      if ( (*(_BYTE *)(v9 - 8) & 1) == 0 && *(_BYTE *)(a1 + 176) == 13 )
      {
        v10 = *(_BYTE *)(v9 + 24);
        if ( v10 == 4 )
        {
          *(_BYTE *)(a1 + 176) = 8;
          v15 = *(_QWORD *)(v9 + 56);
          *(_QWORD *)(a1 + 184) = v15;
          v12 = *(_BYTE *)(v15 - 8) & 1;
        }
        else if ( v10 > 4u )
        {
          if ( v10 != 20 )
            return;
          *(_BYTE *)(a1 + 176) = 11;
          v13 = *(_QWORD *)(v9 + 56);
          *(_QWORD *)(a1 + 184) = v13;
          v12 = *(_BYTE *)(v13 - 8) & 1;
        }
        else if ( v10 == 2 )
        {
          *(_BYTE *)(a1 + 176) = 2;
          v14 = *(_QWORD *)(v9 + 56);
          *(_QWORD *)(a1 + 184) = v14;
          v12 = *(_BYTE *)(v14 - 8) & 1;
        }
        else
        {
          if ( v10 != 3 )
            return;
          *(_BYTE *)(a1 + 176) = 7;
          v11 = *(_QWORD *)(v9 + 56);
          *(_QWORD *)(a1 + 184) = v11;
          v12 = *(_BYTE *)(v11 - 8) & 1;
        }
        if ( v12 )
          *(_DWORD *)(a1 + 192) = 0;
      }
    }
  }
}
