// Function: sub_3508B80
// Address: 0x3508b80
//
__int64 __fastcall sub_3508B80(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 i; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 result; // rax
  __int64 v16; // r14
  _QWORD **v17; // r15
  __int64 v18; // rdx
  unsigned __int16 *v19; // r11
  __int64 v20; // rdi
  _QWORD *v21; // rsi
  _WORD *v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-138h]
  __int64 v24; // [rsp+10h] [rbp-130h]
  __int64 v25; // [rsp+18h] [rbp-128h]
  __int64 v26; // [rsp+20h] [rbp-120h]
  __int64 v27; // [rsp+28h] [rbp-118h]
  __int64 (__fastcall *v28)(__int64); // [rsp+30h] [rbp-110h]
  __int64 v29; // [rsp+38h] [rbp-108h]
  __int64 v30[8]; // [rsp+40h] [rbp-100h] BYREF
  _QWORD v31[24]; // [rsp+80h] [rbp-C0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a2 + 24);
  v5 = v4 + 48;
  for ( i = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 32LL) + 40LL * (*(_DWORD *)(*(_QWORD *)(v4 + 56) + 40LL) & 0xFFFFFF);
        (*(_BYTE *)(v2 + 44) & 4) != 0;
        v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL )
  {
    ;
  }
  while ( 1 )
  {
    v7 = *(_QWORD *)(v2 + 32);
    v8 = v7 + 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
    if ( v7 != v8 )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v5 == v2 )
      break;
    if ( (*(_BYTE *)(v2 + 44) & 4) == 0 )
    {
      v2 = v5;
      break;
    }
  }
  v30[3] = v8;
  v30[1] = v5;
  v30[4] = v5;
  v30[5] = v5;
  v30[6] = i;
  v30[7] = i;
  v30[0] = v2;
  v30[2] = v7;
  sub_3508760(v31, v30, (unsigned __int8 (__fastcall *)(__int64))sub_3507B50);
  v12 = v31[0];
  v13 = v31[1];
  v29 = v31[4];
  v14 = v31[3];
  v27 = v31[6];
  v25 = v31[7];
  v28 = (__int64 (__fastcall *)(__int64))v31[8];
  v26 = v31[9];
  v24 = v31[11];
  v23 = v31[12];
  result = a1;
  v16 = v31[2];
  v17 = (_QWORD **)result;
LABEL_8:
  if ( v12 != v26 )
    goto LABEL_9;
LABEL_29:
  if ( v16 != v24 )
  {
    if ( v16 != v14 || (v9 = v23, v24 != v23) )
    {
      while ( 1 )
      {
LABEL_9:
        if ( *(_BYTE *)v16 == 12 )
        {
          result = sub_3507C70(v17, v16, 0, v9, v10, v11);
        }
        else if ( (*(_BYTE *)(v16 + 3) & 0x10) != 0 )
        {
          result = (__int64)sub_E922F0(*v17, *(_DWORD *)(v16 + 8));
          v9 = result + 2 * v18;
          v19 = (unsigned __int16 *)result;
          if ( result != v9 )
          {
            do
            {
              v11 = (__int64)v17[2];
              v20 = *v19;
              result = *((unsigned __int8 *)v17[6] + v20);
              v10 = (unsigned int)v11;
              if ( (unsigned int)result < (unsigned int)v11 )
              {
                v21 = v17[1];
                while ( 1 )
                {
                  v22 = (_WORD *)v21 + (unsigned int)result;
                  if ( (_WORD)v20 == *v22 )
                    break;
                  result = (unsigned int)(result + 256);
                  if ( (unsigned int)v11 <= (unsigned int)result )
                    goto LABEL_44;
                }
                result = 2 * v11;
                if ( v22 != (_WORD *)((char *)v21 + 2 * v11) )
                {
                  result = (__int64)v21 + result - 2;
                  if ( v22 != (_WORD *)result )
                  {
                    *v22 = *(_WORD *)result;
                    result = (__int64)v17[1];
                    *((_BYTE *)v17[6] + *(unsigned __int16 *)(result + 2LL * (_QWORD)v17[2] - 2)) = ((__int64)v22 - result) >> 1;
                    v11 = (__int64)v17[2];
                  }
                  v17[2] = (_QWORD *)--v11;
                }
              }
LABEL_44:
              ++v19;
            }
            while ( (unsigned __int16 *)v9 != v19 );
          }
        }
        for ( v16 += 40; v16 == v14; v14 = v16 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF) )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 )
            break;
          if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
          {
            v12 = v13;
            break;
          }
          v16 = *(_QWORD *)(v12 + 32);
          result = 5LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
        }
        if ( v12 == v29 )
          goto LABEL_25;
        while ( 1 )
        {
          do
          {
            result = v28(v16);
            if ( (_BYTE)result )
              goto LABEL_8;
            v16 += 40;
            result = v14;
            if ( v16 == v14 )
            {
              while ( 1 )
              {
                v12 = *(_QWORD *)(v12 + 8);
                if ( v13 == v12 )
                {
LABEL_23:
                  v16 = v14;
                  v14 = result;
                  goto LABEL_24;
                }
                if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
                  break;
                v14 = *(_QWORD *)(v12 + 32);
                result = v14 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
                if ( v14 != result )
                  goto LABEL_23;
              }
              v16 = v14;
              v12 = v13;
              v14 = result;
            }
LABEL_24:
            ;
          }
          while ( v12 != v29 );
LABEL_25:
          if ( v16 == v27 )
            break;
          if ( v14 == v16 )
          {
            v9 = v25;
            if ( v27 == v25 )
              break;
          }
        }
        v12 = v29;
        if ( v29 == v26 )
          goto LABEL_29;
      }
    }
  }
  return result;
}
