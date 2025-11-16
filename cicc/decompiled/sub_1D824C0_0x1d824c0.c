// Function: sub_1D824C0
// Address: 0x1d824c0
//
__int64 __fastcall sub_1D824C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r9
  int v6; // r14d
  __int16 *v7; // rdx
  __int16 v8; // ax
  __int16 v10; // ax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // r11
  __int64 *v18; // rax
  __int16 v19; // ax
  __int64 v20; // rax
  __int64 v21; // rdx
  _WORD *v22; // rax
  _WORD *v23; // rdi
  unsigned __int64 v24; // rcx
  _WORD *v25; // rdx
  _QWORD *v26; // rax
  int v27; // eax
  __int64 *v28; // rsi
  unsigned int v29; // edi
  __int64 *v30; // rcx
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  char v40[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v3 = *(_QWORD *)(a2 + 32);
  v4 = sub_1DD5EE0(a2);
  if ( v3 == v4 )
    return 1;
  v5 = v4;
  v6 = 0;
  while ( 1 )
  {
    v7 = *(__int16 **)(v3 + 16);
    v8 = *v7;
    if ( (unsigned __int16)(*v7 - 12) > 1u )
      break;
LABEL_4:
    if ( (*(_BYTE *)v3 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v3 + 46) & 8) != 0 )
        v3 = *(_QWORD *)(v3 + 8);
    }
    v3 = *(_QWORD *)(v3 + 8);
    if ( v3 == v5 )
      return 1;
  }
  if ( (++v6 <= (unsigned int)dword_4FC34E0 || byte_4FC3400)
    && v8 != 45
    && v8
    && (v8 != 1 || (*(_BYTE *)(*(_QWORD *)(v3 + 32) + 64LL) & 8) == 0) )
  {
    v10 = *(_WORD *)(v3 + 46);
    if ( (v10 & 4) != 0 || (v10 & 8) == 0 )
    {
      v11 = (*((_QWORD *)v7 + 1) >> 16) & 1LL;
    }
    else
    {
      v39 = v5;
      LOBYTE(v11) = sub_1E15D00(v3, 0x10000, 1);
      v5 = v39;
    }
    v35 = v5;
    if ( !(_BYTE)v11 )
    {
      v40[0] = 1;
      if ( (unsigned __int8)sub_1E17B50(v3, 0, v40) )
      {
        v12 = *(_QWORD *)(v3 + 32);
        v5 = v35;
        v13 = v12 + 40LL * *(unsigned int *)(v3 + 40);
        if ( v12 == v13 )
          goto LABEL_4;
        while ( 1 )
        {
          if ( *(_BYTE *)v12 == 12 )
            return 0;
          if ( !*(_BYTE *)v12 )
          {
            v14 = *(unsigned int *)(v12 + 8);
            if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 && (int)v14 > 0 )
            {
              v21 = *(_QWORD *)(a1 + 8);
              if ( !v21 )
                BUG();
              v24 = (unsigned int)v14 * (*(_DWORD *)(*(_QWORD *)(v21 + 8) + 24LL * (unsigned int)v14 + 16) & 0xF);
              v22 = (_WORD *)(*(_QWORD *)(v21 + 56)
                            + 2LL * (*(_DWORD *)(*(_QWORD *)(v21 + 8) + 24LL * (unsigned int)v14 + 16) >> 4));
              v23 = v22 + 1;
              LOWORD(v24) = *v22 + v14 * (*(_WORD *)(*(_QWORD *)(v21 + 8) + 24LL * (unsigned int)v14 + 16) & 0xF);
              while ( 1 )
              {
                v25 = v23;
                if ( !v23 )
                  break;
                while ( 1 )
                {
                  ++v25;
                  v26 = (_QWORD *)(*(_QWORD *)(a1 + 608) + ((v24 >> 3) & 0x1FF8));
                  *v26 |= 1LL << v24;
                  v27 = (unsigned __int16)*(v25 - 1);
                  v23 = 0;
                  if ( !(_WORD)v27 )
                    break;
                  v24 = (unsigned int)(v27 + v24);
                  if ( !v25 )
                    goto LABEL_26;
                }
              }
            }
LABEL_26:
            v15 = *(_BYTE *)(v12 + 4);
            if ( (v15 & 1) == 0
              && (v15 & 2) == 0
              && ((*(_BYTE *)(v12 + 3) & 0x10) == 0 || (*(_DWORD *)v12 & 0xFFF00) != 0)
              && (int)v14 < 0 )
            {
              v32 = v5;
              v36 = v13;
              v16 = sub_1E69D00(*(_QWORD *)(a1 + 16), v14);
              v13 = v36;
              v5 = v32;
              v17 = v16;
              if ( v16 )
              {
                if ( *(_QWORD *)(v16 + 24) == *(_QWORD *)(a1 + 24) )
                  break;
              }
            }
          }
LABEL_39:
          v12 += 40;
          if ( v13 == v12 )
            goto LABEL_4;
        }
        v18 = *(__int64 **)(a1 + 512);
        if ( *(__int64 **)(a1 + 520) == v18 )
        {
          v28 = &v18[*(unsigned int *)(a1 + 532)];
          v29 = *(_DWORD *)(a1 + 532);
          if ( v18 != v28 )
          {
            v30 = 0;
            while ( v17 != *v18 )
            {
              if ( *v18 == -2 )
                v30 = v18;
              if ( v28 == ++v18 )
              {
                if ( !v30 )
                  goto LABEL_58;
                *v30 = v17;
                --*(_DWORD *)(a1 + 536);
                ++*(_QWORD *)(a1 + 504);
                break;
              }
            }
LABEL_35:
            v19 = *(_WORD *)(v17 + 46);
            if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
            {
              v20 = (*(_QWORD *)(*(_QWORD *)(v17 + 16) + 8LL) >> 6) & 1LL;
            }
            else
            {
              v34 = v5;
              v38 = v13;
              LOBYTE(v20) = sub_1E15D00(v17, 64, 1);
              v13 = v38;
              v5 = v34;
            }
            if ( (_BYTE)v20 )
              return 0;
            goto LABEL_39;
          }
LABEL_58:
          if ( v29 < *(_DWORD *)(a1 + 528) )
          {
            *(_DWORD *)(a1 + 532) = v29 + 1;
            *v28 = v17;
            ++*(_QWORD *)(a1 + 504);
            goto LABEL_35;
          }
        }
        v31 = v32;
        v33 = v36;
        v37 = v17;
        sub_16CCBA0(a1 + 504, v17);
        v5 = v31;
        v13 = v33;
        v17 = v37;
        goto LABEL_35;
      }
    }
  }
  return 0;
}
