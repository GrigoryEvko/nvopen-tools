// Function: sub_185B7E0
// Address: 0x185b7e0
//
void __fastcall sub_185B7E0(__int64 a1, __int64 a2)
{
  __int64 i; // r15
  __int64 v4; // rcx
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+8h] [rbp-68h]
  _QWORD *v18; // [rsp+8h] [rbp-68h]
  _QWORD v19[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(a1 + 8) )
  {
    v4 = (__int64)sub_1648700(i);
    v5 = *(_BYTE *)(v4 + 16);
    switch ( v5 )
    {
      case '7':
        v6 = *(_QWORD *)(v4 - 24);
        v7 = v4;
        if ( !v6 || v6 != a2 )
          goto LABEL_5;
        break;
      case 'M':
        v7 = v4;
        if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
          v11 = *(_QWORD *)(v4 - 8);
        else
          v11 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
        v4 = sub_157EBA0(*(_QWORD *)(v11
                                   + 0xFFFFFFFD55555558LL * (unsigned int)((i - v11) >> 3)
                                   + 24LL * *(unsigned int *)(v4 + 56)
                                   + 8));
LABEL_5:
        v16 = v4;
        v19[0] = sub_1649960(a2);
        v21 = 773;
        v19[1] = v8;
        v20[0] = v19;
        v20[1] = ".val";
        v9 = sub_1648A60(64, 1u);
        v10 = (__int64)v9;
        if ( v9 )
          sub_15F90E0((__int64)v9, a2, (__int64)v20, v16);
        sub_1648780(v7, a1, v10);
        continue;
      case 'G':
        v18 = (_QWORD *)v4;
        sub_185B7E0(v4, a2);
        sub_15F20C0(v18);
        continue;
      default:
        v7 = v4;
        if ( v5 != 56 )
          goto LABEL_5;
        v17 = v4;
        v12 = sub_15FA1F0(v4);
        v4 = v17;
        if ( !v12 )
          goto LABEL_5;
        v13 = *(_QWORD *)(v17 + 8);
        if ( !v13 )
          goto LABEL_5;
        if ( *(_QWORD *)(v13 + 8) )
          goto LABEL_5;
        v14 = sub_1648700(v13);
        v4 = v17;
        if ( *((_BYTE *)v14 + 16) != 55 )
          goto LABEL_5;
        v15 = *(v14 - 3);
        if ( a2 != v15 || !v15 )
          goto LABEL_5;
        sub_185B7E0(v17, a2);
        v4 = v17;
        break;
    }
    sub_15F20C0((_QWORD *)v4);
  }
}
