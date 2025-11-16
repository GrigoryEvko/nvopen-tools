// Function: sub_E02190
// Address: 0xe02190
//
void __fastcall sub_E02190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // rbx
  __int64 v10; // r12
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rcx
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rcx
  int v21; // esi
  _QWORD *v22; // r11
  unsigned int v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rcx
  _QWORD *v30; // [rsp+8h] [rbp-B8h]
  __int64 v31; // [rsp+10h] [rbp-B0h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  _BYTE *v35; // [rsp+40h] [rbp-80h] BYREF
  __int64 v36; // [rsp+48h] [rbp-78h]
  _BYTE v37[112]; // [rsp+50h] [rbp-70h] BYREF

  for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v10 = *(_QWORD *)(i + 24);
    v11 = *(_BYTE *)v10;
    if ( *(_BYTE *)v10 > 0x1Cu )
    {
      switch ( v11 )
      {
        case 'N':
          sub_E02190(a1, a2, *(_QWORD *)(i + 24), a4, a5, a6);
          break;
        case '=':
          sub_E02020(a2, 0, *(_QWORD *)(v10 + 16), a4, a5, a6);
          break;
        case '?':
          v12 = *(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
          if ( a3 == v12 && v12 && (unsigned __int8)sub_B4DD90(*(_QWORD *)(i + 24)) )
          {
            v14 = 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
            {
              v15 = *(_QWORD *)(v10 - 8);
              v16 = v15 + v14;
            }
            else
            {
              v16 = v10;
              v15 = v10 - v14;
            }
            v17 = v14 - 32;
            v18 = (_QWORD *)(v15 + 32);
            v35 = v37;
            v36 = 0x800000000LL;
            v19 = v17 >> 5;
            if ( (unsigned __int64)v17 > 0x100 )
            {
              v30 = v18;
              v31 = v16;
              v32 = v17 >> 5;
              sub_C8D5F0((__int64)&v35, v37, v19, 8u, v13, v16);
              v22 = v35;
              v21 = v36;
              LODWORD(v19) = v32;
              v16 = v31;
              v18 = v30;
              v20 = &v35[8 * (unsigned int)v36];
            }
            else
            {
              v20 = v37;
              v21 = 0;
              v22 = v37;
            }
            if ( (_QWORD *)v16 != v18 )
            {
              do
              {
                if ( v20 )
                  *v20 = *v18;
                v18 += 4;
                ++v20;
              }
              while ( (_QWORD *)v16 != v18 );
              v22 = v35;
              v21 = v36;
            }
            v23 = v21 + v19;
            v24 = *(_QWORD *)(v10 + 72);
            LODWORD(v36) = v23;
            v25 = sub_AE54E0(a1 + 312, v24, v22, v23);
            sub_E02190(a1, a2, v10, v25 + a4, a5, a6);
            if ( v35 != v37 )
              _libc_free(v35, a2);
          }
          break;
        default:
          if ( v11 == 85 && (unsigned int)sub_B49240(*(_QWORD *)(i + 24)) == 214 )
          {
            v26 = *(_QWORD *)(v10 + 32 * (1LL - (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)));
            if ( *(_BYTE *)v26 == 17 )
            {
              v27 = *(_QWORD **)(v26 + 24);
              v28 = *(_DWORD *)(v26 + 32);
              if ( v28 > 0x40 )
              {
                v29 = *v27 + a4;
              }
              else
              {
                v29 = a4;
                if ( v28 )
                  v29 = ((__int64)((_QWORD)v27 << (64 - (unsigned __int8)v28)) >> (64 - (unsigned __int8)v28)) + a4;
              }
              sub_E02020(a2, 0, *(_QWORD *)(v10 + 16), v29, a5, a6);
            }
          }
          break;
      }
    }
  }
}
