// Function: sub_BD2ED0
// Address: 0xbd2ed0
//
__int64 __fastcall sub_BD2ED0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r8d
  __int64 v5; // rcx
  __int64 v6; // rdx
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  __int64 v16[8]; // [rsp+10h] [rbp-40h] BYREF

  v15 = a2;
  if ( a2 == a3 )
  {
    return 0;
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( v4 )
    {
      v5 = a2;
      v6 = 0;
      v7 = 0;
      while ( 1 )
      {
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        {
          v8 = *(_QWORD *)(a1 - 8) + 32 * v6;
          if ( *(_QWORD *)v8 != v5 )
            goto LABEL_5;
        }
        else
        {
          v8 = a1 + 32 * (v6 - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          if ( v5 != *(_QWORD *)v8 )
            goto LABEL_5;
        }
        if ( v5 )
        {
          v9 = *(_QWORD *)(v8 + 8);
          **(_QWORD **)(v8 + 16) = v9;
          if ( v9 )
            *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
        }
        *(_QWORD *)v8 = a3;
        v7 = 1;
        if ( !a3 )
        {
LABEL_5:
          if ( v4 == (_DWORD)++v6 )
            goto LABEL_16;
          goto LABEL_6;
        }
        v10 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(v8 + 8) = v10;
        if ( v10 )
          *(_QWORD *)(v10 + 16) = v8 + 8;
        ++v6;
        *(_QWORD *)(v8 + 16) = a3 + 16;
        v7 = 1;
        *(_QWORD *)(a3 + 16) = v8;
        if ( v4 == (_DWORD)v6 )
          goto LABEL_16;
LABEL_6:
        v5 = v15;
      }
    }
    v7 = 0;
LABEL_16:
    v11 = sub_BD29A0(a1);
    v12 = v11;
    if ( v11 )
    {
      sub_B58E30(v16, v11);
      LOBYTE(v13) = sub_BD2E50(v16, &v15);
      if ( (_BYTE)v13 )
      {
        v7 = v13;
        sub_B59720(v12, v15, (unsigned __int8 *)a3);
      }
    }
  }
  return v7;
}
