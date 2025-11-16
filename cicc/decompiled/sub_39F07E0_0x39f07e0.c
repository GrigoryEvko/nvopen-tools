// Function: sub_39F07E0
// Address: 0x39f07e0
//
void __fastcall sub_39F07E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rbx
  char v5; // cl
  __int64 v6; // rax
  void *v8; // rsi
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  _QWORD v19[2]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v20[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v21; // [rsp+20h] [rbp-70h]
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-50h]
  _QWORD v24[2]; // [rsp+50h] [rbp-40h] BYREF
  __int16 v25; // [rsp+60h] [rbp-30h]

  v3 = *a2;
  v4 = *(_QWORD *)(*a2 + 24);
  v5 = *(_BYTE *)(v4 + 8);
  if ( (v5 & 1) != 0 )
  {
    v6 = *(_QWORD *)v4;
    v8 = (void *)(*(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v8 )
    {
      if ( (*(_BYTE *)(v4 + 9) & 0xC) != 8 )
      {
LABEL_4:
        v9 = *(_QWORD *)(a1 + 8);
        if ( (v6 & 4) != 0 )
        {
          v10 = *(__int64 **)(v4 - 8);
          v11 = *v10;
          v12 = v10 + 2;
        }
        else
        {
          v11 = 0;
          v12 = 0;
        }
        v19[1] = v11;
        v21 = 771;
        v19[0] = v12;
        v22[0] = v20;
        v22[1] = v19;
        v20[0] = "Reference to undefined temporary symbol ";
        v20[1] = "`";
        v23 = 1282;
        v24[0] = v22;
        v24[1] = "`";
        v25 = 770;
        sub_38BE3D0(v9, *(_QWORD *)(v3 + 8), (__int64)v24);
        return;
      }
      v15 = *(_QWORD *)(v4 + 24);
      *(_BYTE *)(v4 + 8) = v5 | 4;
      v16 = (unsigned __int64)sub_38CE440(v15);
      v6 = v16 | *(_QWORD *)v4 & 7LL;
      *(_QWORD *)v4 = v6;
      if ( !v16 )
        goto LABEL_17;
      v8 = (void *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        if ( (*(_BYTE *)(v4 + 9) & 0xC) != 8 )
        {
          if ( !off_4CF6DB8 )
            goto LABEL_17;
LABEL_22:
          BUG();
        }
        *(_BYTE *)(v4 + 8) |= 4u;
        v17 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v4 + 24));
        v18 = v17 | *(_QWORD *)v4 & 7LL;
        *(_QWORD *)v4 = v18;
        LOBYTE(v6) = v18;
        if ( off_4CF6DB8 == (_UNKNOWN *)v17 )
          goto LABEL_17;
        v13 = v18 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v13 )
        {
          if ( (*(_BYTE *)(v4 + 9) & 0xC) != 8 )
            goto LABEL_22;
          *(_BYTE *)(v4 + 8) |= 4u;
          v13 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v4 + 24));
          *(_QWORD *)v4 = v13 | *(_QWORD *)v4 & 7LL;
          if ( !v13 )
            goto LABEL_22;
        }
LABEL_10:
        v14 = *(_QWORD *)(*(_QWORD *)(v13 + 24) + 8LL);
        *(_BYTE *)(v14 + 9) |= 2u;
        *a2 = sub_38CF310(v14, *(_WORD *)(*a2 + 16), *(_QWORD *)(a1 + 8), *(_QWORD *)(*a2 + 8));
        return;
      }
    }
    v13 = v6 & 0xFFFFFFFFFFFFFFF8LL;
    if ( off_4CF6DB8 != v8 )
      goto LABEL_10;
LABEL_17:
    v3 = *a2;
    goto LABEL_4;
  }
  sub_390D5F0(*(_QWORD *)(a1 + 264), *(_QWORD *)(*a2 + 24), (bool *)v24);
  if ( LOBYTE(v24[0]) )
  {
    sub_38E2920(v4, 2u);
    *(_BYTE *)(v4 + 8) |= 0x10u;
  }
}
