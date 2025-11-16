// Function: sub_2BFD6A0
// Address: 0x2bfd6a0
//
__int64 __fastcall sub_2BFD6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v4; // eax
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rdi
  int v12; // eax
  __int64 v13; // rsi
  unsigned __int8 v14; // al
  __int64 v15; // r8
  unsigned int v16; // esi
  __int64 v17; // r8
  int v18; // r11d
  __int64 *v19; // rcx
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  int v27; // eax
  __int64 v28; // r8
  unsigned int v29; // esi
  int v30; // edx
  __int64 v31; // rdi
  int v32; // r8d
  int v33; // eax
  int v34; // eax
  int v35; // esi
  __int64 v36; // rdi
  int v37; // r9d
  unsigned int v38; // r14d
  __int64 *v39; // r8
  __int64 v40; // rax
  int v41; // r10d
  __int64 *v42; // r9

  v4 = *(_DWORD *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  if ( v4 )
  {
    v7 = v4 - 1;
    v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      v2 = v9[1];
      if ( v2 )
        return v2;
    }
    else
    {
      v12 = 1;
      while ( v10 != -4096 )
      {
        v32 = v12 + 1;
        v8 = v7 & (v12 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_3;
        v12 = v32;
      }
    }
  }
  if ( sub_2BF04A0(a2) )
  {
    v13 = sub_2BF04A0(a2);
    v14 = *(_BYTE *)(v13 + 8);
    switch ( v14 )
    {
      case 0x1Eu:
      case 0x1Du:
      case 0x20u:
      case 0x24u:
      case 0x22u:
      case 0x1Fu:
      case 0x23u:
        v15 = 0;
        if ( *(_DWORD *)(v13 + 56) )
          v15 = **(_QWORD **)(v13 + 48);
        v2 = sub_2BFD6A0(a1, v15);
        break;
      case 0x21u:
        v25 = *(_QWORD *)(v13 + 160);
        if ( v25 )
        {
LABEL_26:
          v2 = *(_QWORD *)(v25 + 8);
          break;
        }
        v2 = *(_QWORD *)(*(_QWORD *)(v13 + 136) + 8LL);
        break;
      case 1u:
        v2 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v13 + 48) + 40LL) + 8LL);
        break;
      default:
        if ( (unsigned int)v14 - 6 <= 1
          || v14 == 28
          || v14 == 27
          || v14 == 11
          || v14 == 17
          || v14 == 12
          || v14 == 13
          || v14 == 15
          || v14 == 8 )
        {
          v2 = sub_2BFD6A0(a1, **(_QWORD **)(v13 + 48));
        }
        else
        {
          switch ( v14 )
          {
            case 0x19u:
              v2 = sub_2BFDBB0(a1, v13);
              break;
            case 4u:
              v2 = sub_2BFE150(a1, v13);
              break;
            case 0x17u:
              v2 = sub_2BFE2A0(a1, v13);
              break;
            case 9u:
              v2 = sub_2BFE840(a1, v13);
              break;
            case 0xEu:
              v2 = sub_2BFCA80(a1, v13);
              break;
            default:
              if ( (unsigned __int8)(v14 - 19) > 3u )
              {
                switch ( v14 )
                {
                  case 0x18u:
                    v2 = sub_2BFE5E0(a1, v13);
                    break;
                  case 0x12u:
                    goto LABEL_54;
                  case 5u:
                    v25 = *(_QWORD *)(a2 + 40);
                    goto LABEL_26;
                  case 0x10u:
LABEL_54:
                    v2 = *(_QWORD *)(v13 + 168);
                    break;
                  case 0xAu:
                    v2 = *(_QWORD *)(v13 + 160);
                    break;
                  case 2u:
                    v2 = sub_D95540(*(_QWORD *)(v13 + 152));
                    break;
                }
              }
              else
              {
                v2 = sub_2BFCA90(a1, v13);
              }
              break;
          }
        }
        break;
    }
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 )
    {
      v17 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      v19 = 0;
      v20 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = (__int64 *)(v17 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == a2 )
      {
LABEL_14:
        v23 = v21 + 1;
LABEL_15:
        *v23 = v2;
        return v2;
      }
      while ( v22 != -4096 )
      {
        if ( !v19 && v22 == -8192 )
          v19 = v21;
        v20 = (v16 - 1) & (v18 + v20);
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == a2 )
          goto LABEL_14;
        ++v18;
      }
      if ( !v19 )
        v19 = v21;
      v33 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v30 = v33 + 1;
      if ( 4 * (v33 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 20) - v30 > v16 >> 3 )
        {
LABEL_65:
          *(_DWORD *)(a1 + 16) = v30;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v19 = a2;
          v23 = v19 + 1;
          v19[1] = 0;
          goto LABEL_15;
        }
        sub_2BFD020(a1, v16);
        v34 = *(_DWORD *)(a1 + 24);
        if ( v34 )
        {
          v35 = v34 - 1;
          v36 = *(_QWORD *)(a1 + 8);
          v37 = 1;
          v38 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v39 = 0;
          v30 = *(_DWORD *)(a1 + 16) + 1;
          v19 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v19;
          if ( *v19 != a2 )
          {
            while ( v40 != -4096 )
            {
              if ( !v39 && v40 == -8192 )
                v39 = v19;
              v38 = v35 & (v37 + v38);
              v19 = (__int64 *)(v36 + 16LL * v38);
              v40 = *v19;
              if ( *v19 == a2 )
                goto LABEL_65;
              ++v37;
            }
            if ( v39 )
              v19 = v39;
          }
          goto LABEL_65;
        }
LABEL_103:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_2BFD020(a1, 2 * v16);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (__int64 *)(v28 + 16LL * v29);
      v31 = *v19;
      if ( *v19 != a2 )
      {
        v41 = 1;
        v42 = 0;
        while ( v31 != -4096 )
        {
          if ( !v42 && v31 == -8192 )
            v42 = v19;
          v29 = v27 & (v41 + v29);
          v19 = (__int64 *)(v28 + 16LL * v29);
          v31 = *v19;
          if ( *v19 == a2 )
            goto LABEL_65;
          ++v41;
        }
        if ( v42 )
          v19 = v42;
      }
      goto LABEL_65;
    }
    goto LABEL_103;
  }
  v24 = *(_QWORD *)(a2 + 40);
  if ( !v24 )
    return *(_QWORD *)(a1 + 32);
  return *(_QWORD *)(v24 + 8);
}
