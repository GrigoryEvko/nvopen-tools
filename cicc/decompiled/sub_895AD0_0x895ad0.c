// Function: sub_895AD0
// Address: 0x895ad0
//
__int64 __fastcall sub_895AD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  char v3; // al
  __int64 v4; // r15
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  char v11; // al
  __int64 v12; // rdx
  __int64 i; // rax
  _QWORD *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 j; // rax
  _QWORD *v18; // rdx
  __int64 *v19; // r12
  int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r9
  __int64 v30; // rax
  char v31; // al
  int v32; // [rsp+4h] [rbp-3Ch]
  __int16 v33; // [rsp+8h] [rbp-38h]
  unsigned int v34; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 88);
  if ( (*(_BYTE *)(v2 + 194) & 0x40) != 0 )
  {
    v8 = *(_QWORD *)(a2 + 48);
    if ( (*(_BYTE *)(v8 + 32) & 0x10) != 0 )
      sub_895AD0(**(_QWORD **)(v2 + 232), *(_QWORD *)(a2 + 48));
    sub_865D70(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL), 1, 0, 1u, 1u, 0);
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_QWORD *)(v8 + 40);
    if ( !v9 || v9 == v10 )
      *(_QWORD *)(a2 + 40) = sub_73BB50(v10);
    v11 = *(_BYTE *)(v8 + 32) & 4 | *(_BYTE *)(a2 + 32) & 0xFB;
    *(_BYTE *)(a2 + 32) = v11;
    *(_QWORD *)(a2 + 56) = *(_QWORD *)(v8 + 56);
    *(_BYTE *)(a2 + 32) = *(_BYTE *)(v8 + 32) & 8 | v11 & 0xF7;
    result = (__int64)sub_866010();
  }
  else
  {
    v3 = *(_BYTE *)(a2 + 32);
    v4 = *(_QWORD *)(a1 + 96);
    if ( (v3 & 0x20) != 0 )
    {
      sub_6851C0(0x3EFu, dword_4F07508);
      result = (__int64)sub_7305E0();
      *(_QWORD *)(a2 + 40) = result;
      *(_BYTE *)(v4 + 81) |= 4u;
      goto LABEL_8;
    }
    if ( (*(_BYTE *)(v4 + 81) & 4) == 0 )
    {
      if ( qword_4F60188 != unk_4D042F0 )
      {
        *(_BYTE *)(a2 + 32) = v3 | 0x20;
        v6 = *(_QWORD *)(v4 + 32);
        switch ( *(_BYTE *)(v6 + 80) )
        {
          case 4:
          case 5:
            v12 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 80LL);
            break;
          case 6:
            v12 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 32LL);
            break;
          case 9:
          case 0xA:
            v12 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 56LL);
            break;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v12 = *(_QWORD *)(v6 + 88);
            break;
          default:
            BUG();
        }
        for ( i = *(_QWORD *)(*(_QWORD *)(v12 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v14 = *(_QWORD **)(v12 + 288);
        for ( result = **(_QWORD **)(i + 168); result; result = *(_QWORD *)result )
        {
          if ( *(_DWORD *)(result + 36) == *(_DWORD *)(a2 + 36) )
            break;
          if ( (*(_BYTE *)(result + 32) & 4) != 0 )
            v14 = (_QWORD *)*v14;
        }
        if ( !v14 )
          goto LABEL_44;
        if ( (*(_BYTE *)(v14[6] + 32LL) & 0x20) == 0 )
        {
          v34 = sub_8D0B70(v6);
          sub_864700(
            v14[5],
            0,
            v2,
            *(_QWORD *)(v4 + 24),
            *(_QWORD *)(v4 + 32),
            *(_QWORD *)(v2 + 240),
            1,
            4 * ((*(_BYTE *)(v2 + 195) & 8) != 0));
          sub_8600D0(1u, -1, *(_QWORD *)(v2 + 152), 0);
          v15 = *(_QWORD *)(v4 + 64);
          if ( v15 )
            sub_886000(v15);
          if ( dword_4F077C4 == 2 )
            *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
          sub_7BC160((__int64)(v14 + 1));
          ++qword_4F60188;
          v32 = dword_4F061D8;
          v33 = unk_4F061DC;
          sub_6794F0((__int64 **)a2, a1, 0);
          --qword_4F60188;
          sub_8CA950(v2, a2);
          dword_4F061D8 = v32;
          v16 = *(_QWORD *)(v4 + 120);
          unk_4F061DC = v33;
          if ( v16 )
          {
            for ( j = *(_QWORD *)(v2 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            v18 = **(_QWORD ***)(j + 168);
            v19 = **(__int64 ***)(v16 + 168);
            if ( v18 && (_QWORD *)a2 != v18 )
            {
              v20 = 1;
              do
              {
                v18 = (_QWORD *)*v18;
                ++v20;
              }
              while ( (_QWORD *)a2 != v18 && v18 );
              do
              {
                --v20;
                v19 = (__int64 *)*v19;
              }
              while ( v20 != 1 );
            }
            if ( !v19[5] )
            {
              v31 = *((_BYTE *)v19 + 32) | 4;
              *((_BYTE *)v19 + 32) = v31;
              *((_BYTE *)v19 + 32) = *(_BYTE *)(a2 + 32) & 8 | v31 & 0xF7;
              v19[5] = (__int64)sub_73BB50(*(_QWORD *)(a2 + 40));
            }
          }
          v21 = v2;
          sub_884800(v2);
          if ( dword_4F077C4 == 2 )
          {
            v23 = qword_4F04C68[0];
            v21 = (int)dword_4F04C40;
            v30 = 776LL * (int)dword_4F04C40;
            *(_BYTE *)(qword_4F04C68[0] + v30 + 7) &= ~8u;
            v22 = qword_4F04C68[0];
            if ( *(_QWORD *)(qword_4F04C68[0] + v30 + 456) )
              sub_8845B0(v21);
          }
          sub_863FC0(v21, a2, v22, v23, v24, v25);
          sub_863FE0(v21, a2, v26, v27, v28, v29);
          result = v34;
          if ( v34 )
            result = sub_8D0B10();
LABEL_44:
          *(_BYTE *)(a2 + 32) &= ~0x20u;
          *(_BYTE *)(v4 + 81) &= ~4u;
          goto LABEL_8;
        }
      }
      sub_6851C0(0x3EFu, dword_4F07508);
    }
    result = (__int64)sub_7305E0();
    *(_QWORD *)(a2 + 40) = result;
  }
LABEL_8:
  *(_BYTE *)(a2 + 32) &= ~0x10u;
  return result;
}
