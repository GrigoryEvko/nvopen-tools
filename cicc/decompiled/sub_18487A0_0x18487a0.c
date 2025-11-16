// Function: sub_18487A0
// Address: 0x18487a0
//
__int64 __fastcall sub_18487A0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // r12
  char v4; // al
  __int64 v5; // r15
  signed __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned __int8 v18; // [rsp+Fh] [rbp-41h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  __int64 *v20; // [rsp+18h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 80);
  v18 = 0;
  v20 = &v2[*(unsigned int *)(a1 + 88)];
  if ( v2 == v20 )
    return v18;
LABEL_4:
  while ( 2 )
  {
    v3 = *v2;
    if ( !sub_15E4F60(*v2) )
    {
      sub_15E4B50(v3);
      if ( !v4 )
      {
        if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL) + 8LL) )
        {
          if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
          {
            sub_15E08E0(v3, a2);
            v5 = *(_QWORD *)(v3 + 88);
            v19 = v5 + 40LL * *(_QWORD *)(v3 + 96);
            if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
            {
              sub_15E08E0(v3, a2);
              v5 = *(_QWORD *)(v3 + 88);
            }
          }
          else
          {
            v5 = *(_QWORD *)(v3 + 88);
            v19 = v5 + 40LL * *(_QWORD *)(v3 + 96);
          }
          a2 = 0xCCCCCCCCCCCCCCCDLL;
          v6 = 0xCCCCCCCCCCCCCCCDLL * ((v19 - v5) >> 3);
          if ( v6 >> 2 > 0 )
          {
            v7 = v5 + 160 * (v6 >> 2);
            while ( !(unsigned __int8)sub_15E0510(v5) )
            {
              v8 = v5;
              v5 += 40;
              if ( (unsigned __int8)sub_15E0510(v5) )
                break;
              v5 = v8 + 80;
              if ( (unsigned __int8)sub_15E0510(v8 + 80) )
                break;
              v5 = v8 + 120;
              if ( (unsigned __int8)sub_15E0510(v8 + 120) )
                break;
              v5 = v8 + 160;
              if ( v7 == v8 + 160 )
              {
                v6 = 0xCCCCCCCCCCCCCCCDLL * ((v19 - v5) >> 3);
                goto LABEL_36;
              }
            }
LABEL_16:
            if ( v19 != v5 )
              goto LABEL_3;
            goto LABEL_17;
          }
LABEL_36:
          if ( v6 != 2 )
          {
            if ( v6 != 3 )
            {
              if ( v6 == 1 && (unsigned __int8)sub_15E0510(v5) )
                goto LABEL_16;
              goto LABEL_17;
            }
            if ( (unsigned __int8)sub_15E0510(v5) )
              goto LABEL_16;
            v5 += 40;
          }
          if ( (unsigned __int8)sub_15E0510(v5) )
            goto LABEL_16;
          v5 += 40;
          if ( (unsigned __int8)sub_15E0510(v5) )
            goto LABEL_16;
LABEL_17:
          v9 = *(_QWORD *)(v3 + 80);
          v10 = v3 + 72;
          if ( v9 == v3 + 72 )
            goto LABEL_3;
          v11 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v12 = v9 - 24;
              if ( !v9 )
                v12 = 0;
              v13 = sub_157EBA0(v12);
              if ( *(_BYTE *)(v13 + 16) == 25 )
                break;
LABEL_20:
              v9 = *(_QWORD *)(v9 + 8);
              if ( v10 == v9 )
                goto LABEL_30;
            }
            v14 = 0;
            v15 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
            if ( v15 )
            {
              a2 = 4LL * v15;
              v14 = *(_QWORD *)(v13 - 24LL * v15);
            }
            v16 = sub_1649C60(v14);
            if ( *(_BYTE *)(v16 + 16) != 17 || *(_QWORD *)v16 != **(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL) )
              break;
            if ( v11 )
            {
              if ( v11 != v16 )
                break;
              goto LABEL_20;
            }
            v9 = *(_QWORD *)(v9 + 8);
            v11 = v16;
            if ( v10 == v9 )
            {
LABEL_30:
              if ( !v11 )
                break;
              a2 = 38;
              ++v2;
              sub_15E0E40(v11, 38);
              v18 = 1;
              if ( v20 == v2 )
                return v18;
              goto LABEL_4;
            }
          }
        }
      }
    }
LABEL_3:
    if ( v20 != ++v2 )
      continue;
    return v18;
  }
}
