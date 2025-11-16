// Function: sub_298B0B0
// Address: 0x298b0b0
//
__int64 sub_298B0B0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int128 *v13; // rax

  v0 = sub_22077B0(0xB0u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_500778C;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_WORD *)(v0 + 168) = 0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    *(_DWORD *)(v0 + 24) = 0;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)v0 = off_4A223E0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    v2 = sub_C52410();
    v3 = v2 + 1;
    v4 = sub_C959E0();
    v5 = (_QWORD *)v2[2];
    if ( v5 )
    {
      v6 = v2 + 1;
      do
      {
        while ( 1 )
        {
          v7 = v5[2];
          v8 = v5[3];
          if ( v4 <= v5[4] )
            break;
          v5 = (_QWORD *)v5[3];
          if ( !v8 )
            goto LABEL_7;
        }
        v6 = v5;
        v5 = (_QWORD *)v5[2];
      }
      while ( v7 );
LABEL_7:
      if ( v3 != v6 && v4 >= v6[4] )
        v3 = v6;
    }
    if ( v3 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v9 = v3[7];
      if ( v9 )
      {
        v10 = v3 + 6;
        do
        {
          while ( 1 )
          {
            v11 = *(_QWORD *)(v9 + 16);
            v12 = *(_QWORD *)(v9 + 24);
            if ( *(_DWORD *)(v9 + 32) >= dword_5007888 )
              break;
            v9 = *(_QWORD *)(v9 + 24);
            if ( !v12 )
              goto LABEL_16;
          }
          v10 = (_QWORD *)v9;
          v9 = *(_QWORD *)(v9 + 16);
        }
        while ( v11 );
LABEL_16:
        if ( v3 + 6 != v10 && dword_5007888 >= *((_DWORD *)v10 + 8) && *((_DWORD *)v10 + 9) )
          *(_BYTE *)(v1 + 169) = qword_5007908;
      }
    }
    v13 = sub_BC2B00();
    sub_298B030((__int64)v13);
  }
  return v1;
}
