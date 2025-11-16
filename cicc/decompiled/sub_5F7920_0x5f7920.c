// Function: sub_5F7920
// Address: 0x5f7920
//
__int64 __fastcall sub_5F7920(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // r13
  char v7; // al
  char v8; // al
  __int64 v9; // rcx
  char v10; // dl
  char v11; // al
  __int64 result; // rax
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // rdi
  _DWORD *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // r12
  _QWORD *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  _QWORD *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // [rsp+8h] [rbp-38h]

  v5 = 0;
  v7 = *((_BYTE *)a3 + 561) & 0xFB;
  *((_BYTE *)a3 + 561) = v7;
  if ( word_4F06418[0] == 55 )
  {
    *((_BYTE *)a3 + 561) = v7 | 4;
    *(_QWORD *)((char *)a3 + 564) = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a1, a2, a3, a4);
    v28 = sub_724DC0(a1, a2, v24, v25, v26, v27);
    a3[72] = v28;
    sub_6C9ED0(0, 0, v28);
    if ( *(_QWORD *)(a3[72] + 144LL) && (*(_BYTE *)(a2 + 9) & 4) != 0 && (unsigned int)sub_731A30() )
    {
      v37 = a3[72];
      v5 = *(_QWORD *)(v37 + 144);
      *(_QWORD *)(v37 + 144) = 0;
    }
    else
    {
      v5 = 0;
    }
    if ( dword_4D043E0 )
      sub_650E40(a3);
    a3[66] = unk_4F061D8;
  }
  v8 = 0;
  if ( unk_4D04424 )
  {
    if ( (unsigned __int16)(word_4F06418[0] - 16) <= 0x39u )
    {
      v9 = 0x200010000000019LL;
      if ( _bittest64(&v9, (unsigned int)word_4F06418[0] - 16) )
        v8 = ((*(_BYTE *)(a1 + 17) >> 5) ^ 1) & 1;
    }
  }
  v10 = 4 * v8;
  v11 = (4 * v8) | *((_BYTE *)a3 + 127) & 0xFB;
  *((_BYTE *)a3 + 127) = v10 | *((_BYTE *)a3 + 127) & 0xFB;
  if ( (v11 & 4) != 0 && (*((_BYTE *)a3 + 561) & 4) != 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 202001) )
  {
    if ( dword_4F077BC )
    {
      if ( !dword_4F077B4 )
      {
        if ( qword_4F077A8 > 0x1387Fu )
        {
LABEL_14:
          sub_684B30(3306, &dword_4F063F8);
          goto LABEL_8;
        }
        goto LABEL_32;
      }
    }
    else if ( !dword_4F077B4 )
    {
LABEL_32:
      *((_BYTE *)a3 + 127) = v11 & 0xFB;
      goto LABEL_8;
    }
    if ( unk_4F077A0 > 0xEA5Fu )
      goto LABEL_14;
    goto LABEL_32;
  }
LABEL_8:
  sub_5F4F20(a1, a2, (__int64)a3, dword_4F04C64);
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(result + 4) == 6 )
  {
    result = *a3;
    if ( *a3 )
    {
      v13 = *(_QWORD *)(result + 88);
      if ( v5 )
      {
        v42 = sub_867B10();
        sub_7296B0(*(unsigned int *)(*(_QWORD *)(v42 + 32) + 164LL), a2, v14, v42);
        v15 = sub_73B8B0(v5, 0x2000);
        sub_7296B0(unk_4F073B8, 0x2000, v16, v17);
        result = sub_72D910(v15, 7, v13);
      }
      if ( (*((_BYTE *)a3 + 127) & 4) != 0 )
      {
        v18 = *(_QWORD *)(**(_QWORD **)a2 + 96LL);
        v19 = *(_QWORD *)(*(_QWORD *)a2 + 168LL);
        if ( *(_BYTE *)(*(_QWORD *)a2 + 140LL) == 11 && (*(_BYTE *)(v19 + 111) & 2) != 0 )
        {
          v38 = *a3;
          v39 = *(__int64 **)(**(_QWORD **)a2 + 96LL);
          sub_5E4DE0(v39, *a3, (__int64)(a3 + 6));
          if ( word_4F06418[0] == 16 )
            return sub_7B8B50(v39, v38, v40, v41);
          else
            return sub_7BE180();
        }
        *(_BYTE *)(v19 + 111) |= 2u;
        *(_BYTE *)(v13 + 145) |= 0x20u;
        ++*(_DWORD *)(v18 + 100);
        v20 = *a3;
        v21 = *(_DWORD **)(*a3 + 104LL);
        *((_BYTE *)a3 + 124) &= ~0x40u;
        if ( (*(_WORD *)(a2 + 8) & 0x180) != 0 && (*(_DWORD *)(*(_QWORD *)a2 + 176LL) & 0x44000) == 0 )
        {
          *v21 = dword_4F06650[0];
        }
        else
        {
          v22 = *(_QWORD *)(a2 + 32);
          if ( v22 )
          {
            if ( word_4F06418[0] == 16 )
            {
              sub_892960();
              *(_BYTE *)(v18 + 183) |= 0x20u;
              sub_7B8B50(v20, v22, v35, v36);
            }
            else
            {
              sub_7BE180();
              v23 = sub_725A70(2);
              *(_QWORD *)(v13 + 152) = v23;
              *(_QWORD *)(v23 + 56) = sub_72C9A0();
              --*(_DWORD *)(v18 + 100);
            }
LABEL_50:
            result = *(unsigned __int8 *)(a2 + 8);
            *(_BYTE *)(a2 + 8) |= 4u;
            if ( !unk_4D0441C )
            {
              result = (unsigned int)result | 6;
              *(_BYTE *)(a2 + 8) = result;
            }
            *(_BYTE *)(a2 + 9) |= 0x10u;
            return result;
          }
        }
        if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) )
          sub_684B40(&dword_4F063F8, 2512);
        v29 = sub_63B4E0(*a3);
        v30 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        v31 = (_QWORD *)qword_4CF7FE0;
        if ( qword_4CF7FE0 )
          qword_4CF7FE0 = *(_QWORD *)qword_4CF7FE0;
        else
          v31 = (_QWORD *)sub_823970(24);
        v31[1] = 0;
        v31[2] = 0;
        *v31 = 0;
        v32 = *a3;
        v31[2] = v29;
        v31[1] = v32;
        v33 = *(_QWORD *)(**(_QWORD **)(v30 + 208) + 96LL);
        for ( *(_BYTE *)(v33 + 183) |= 0x40u; *(_BYTE *)(v30 - 772) == 6; v30 -= 776 )
          ;
        v34 = *(_QWORD **)(v30 + 272);
        if ( v34 )
          *v34 = v31;
        else
          *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v30 + 208) + 96LL) + 64LL) = v31;
        *(_QWORD *)(v30 + 272) = v31;
        goto LABEL_50;
      }
    }
  }
  return result;
}
