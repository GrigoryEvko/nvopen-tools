// Function: sub_38CF7C0
// Address: 0x38cf7c0
//
void __fastcall sub_38CF7C0(unsigned __int64 a1)
{
  char v2; // al
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == -1 )
  {
LABEL_6:
    nullsub_1930();
    j_j___libc_free_0(a1);
  }
  else
  {
    switch ( v2 )
    {
      case 0:
      case 5:
      case 10:
      case 13:
        goto LABEL_6;
      case 1:
        v16 = *(_QWORD *)(a1 + 112);
        if ( v16 != a1 + 128 )
          _libc_free(v16);
        v17 = *(_QWORD *)(a1 + 64);
        if ( v17 != a1 + 80 )
          _libc_free(v17);
        goto LABEL_42;
      case 2:
      case 8:
        v3 = *(_QWORD *)(a1 + 64);
        if ( v3 != a1 + 80 )
          _libc_free(v3);
        goto LABEL_42;
      case 3:
        goto LABEL_42;
      case 4:
        v5 = *(_QWORD *)(a1 + 144);
        if ( v5 != a1 + 160 )
          _libc_free(v5);
        v6 = *(_QWORD *)(a1 + 88);
        if ( v6 != a1 + 104 )
          _libc_free(v6);
        v7 = *(_QWORD *)(a1 + 64);
        if ( v7 != a1 + 80 )
          _libc_free(v7);
        goto LABEL_42;
      case 6:
        v14 = *(_QWORD *)(a1 + 88);
        if ( v14 != a1 + 104 )
          _libc_free(v14);
        v15 = *(_QWORD *)(a1 + 64);
        if ( v15 != a1 + 80 )
          _libc_free(v15);
        goto LABEL_42;
      case 7:
        v4 = *(_QWORD *)(a1 + 56);
        if ( v4 != a1 + 72 )
          _libc_free(v4);
        goto LABEL_42;
      case 9:
        v8 = *(_QWORD *)(a1 + 96);
        if ( v8 != a1 + 112 )
          _libc_free(v8);
        goto LABEL_42;
      case 11:
        v13 = *(_QWORD *)(a1 + 80);
        if ( v13 != a1 + 96 )
          _libc_free(v13);
        goto LABEL_42;
      case 12:
        v9 = *(_QWORD *)(a1 + 272);
        if ( v9 != a1 + 288 )
          _libc_free(v9);
        v10 = *(_QWORD *)(a1 + 224);
        if ( v10 != a1 + 240 )
          _libc_free(v10);
        v11 = *(_QWORD *)(a1 + 112);
        if ( v11 != a1 + 128 )
          _libc_free(v11);
        v12 = *(_QWORD *)(a1 + 64);
        if ( v12 != a1 + 80 )
          _libc_free(v12);
LABEL_42:
        nullsub_1930();
        j_j___libc_free_0(a1);
        break;
      default:
        return;
    }
  }
}
