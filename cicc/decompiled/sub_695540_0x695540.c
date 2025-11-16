// Function: sub_695540
// Address: 0x695540
//
void __fastcall sub_695540(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r8

  if ( (*(_BYTE *)(a1 + 195) & 8) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 207) & 0x20) != 0 )
      v4 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 760);
    else
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 160LL);
    while ( *(_BYTE *)(v4 + 140) == 12 )
      v4 = *(_QWORD *)(v4 + 160);
    LODWORD(v9) = sub_8D3EA0(v4);
    if ( (_DWORD)v9 )
    {
      v10 = sub_72CBE0(v4, a2, v5, v6, v7, v8);
      sub_65C2A0(v4, v10);
      v9 = *(_DWORD *)(*(_QWORD *)(v4 + 168) + 24LL) == 2;
      if ( !(_DWORD)a2 )
      {
LABEL_8:
        sub_68A860(v10, (__int64)a3, *(_QWORD *)(unk_4F04C50 + 32LL), v11, v12);
        return;
      }
    }
    else
    {
      sub_6851C0(0x633u, a3);
      v10 = sub_72C930(1587);
      if ( !(_DWORD)a2 )
        goto LABEL_8;
    }
    v10 = sub_72B740(v10, (unsigned int)v9);
    goto LABEL_8;
  }
}
