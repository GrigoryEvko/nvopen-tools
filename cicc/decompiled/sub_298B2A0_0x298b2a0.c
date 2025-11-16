// Function: sub_298B2A0
// Address: 0x298b2a0
//
__int64 __fastcall sub_298B2A0(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int128 *v14; // rax

  v1 = sub_22077B0(0xB0u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_500778C;
    *(_QWORD *)(v1 + 56) = v1 + 104;
    *(_QWORD *)(v1 + 112) = v1 + 160;
    *(_BYTE *)(v1 + 169) = a1;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    *(_DWORD *)(v1 + 24) = 0;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)v1 = off_4A223E0;
    v3 = sub_C52410();
    v4 = v3 + 1;
    v5 = sub_C959E0();
    v6 = (_QWORD *)v3[2];
    if ( v6 )
    {
      v7 = v3 + 1;
      do
      {
        while ( 1 )
        {
          v8 = v6[2];
          v9 = v6[3];
          if ( v5 <= v6[4] )
            break;
          v6 = (_QWORD *)v6[3];
          if ( !v9 )
            goto LABEL_7;
        }
        v7 = v6;
        v6 = (_QWORD *)v6[2];
      }
      while ( v8 );
LABEL_7:
      if ( v4 != v7 && v5 >= v7[4] )
        v4 = v7;
    }
    if ( v4 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v10 = v4[7];
      if ( v10 )
      {
        v11 = v4 + 6;
        do
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(v10 + 16);
            v13 = *(_QWORD *)(v10 + 24);
            if ( *(_DWORD *)(v10 + 32) >= dword_5007888 )
              break;
            v10 = *(_QWORD *)(v10 + 24);
            if ( !v13 )
              goto LABEL_16;
          }
          v11 = (_QWORD *)v10;
          v10 = *(_QWORD *)(v10 + 16);
        }
        while ( v12 );
LABEL_16:
        if ( v4 + 6 != v11 && dword_5007888 >= *((_DWORD *)v11 + 8) && *((_DWORD *)v11 + 9) )
          *(_BYTE *)(v2 + 169) = qword_5007908;
      }
    }
    v14 = sub_BC2B00();
    sub_298B030((__int64)v14);
  }
  return v2;
}
