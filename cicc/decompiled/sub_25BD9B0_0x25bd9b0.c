// Function: sub_25BD9B0
// Address: 0x25bd9b0
//
__int64 __fastcall sub_25BD9B0(__int64 *a1, unsigned __int8 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  unsigned int v5; // eax
  int v6; // edx
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  unsigned __int16 v11; // cx
  int v12; // eax
  _QWORD *v13; // rdi
  _QWORD *v14; // rsi
  __int64 v15; // rdi
  unsigned int v16; // r14d
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // rsi
  __int64 v21; // rdi
  int v22; // esi
  int v23; // r9d
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  LOBYTE(v2) = sub_B46560(a2);
  v3 = v2;
  if ( !(_BYTE)v2 )
  {
    LOBYTE(v5) = sub_B46500(a2);
    v6 = *a2;
    if ( (_BYTE)v5 )
    {
      if ( (_BYTE)v6 == 64 )
      {
        LOBYTE(v3) = a2[72] != 0;
        return v3;
      }
      if ( (unsigned __int8)(v6 - 65) <= 1u )
        return v5;
      if ( (_BYTE)v6 != 62 && (_BYTE)v6 != 61 )
        BUG();
      v11 = *((_WORD *)a2 + 1);
      if ( ((v11 >> 7) & 6) != 0 )
        return v5;
      v7 = (unsigned int)(v6 - 34);
      v5 = v11 & 1;
      if ( (_BYTE)v5 )
        return v5;
    }
    else
    {
      v7 = (unsigned int)(v6 - 34);
      if ( (unsigned __int8)v7 > 0x33u )
        return v3;
    }
    v8 = 0x8000000000041LL;
    if ( _bittest64(&v8, v7) )
    {
      v9 = *a1;
      if ( !(unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 39) && !(unsigned __int8)sub_B49560((__int64)a2, 39) )
      {
        v10 = *((_QWORD *)a2 - 4);
        if ( *a2 == 85 )
        {
          if ( !v10 || *(_BYTE *)v10 )
            return 1;
          v24 = *(_QWORD *)(v10 + 24);
          if ( v24 == *((_QWORD *)a2 + 10)
            && (*(_BYTE *)(v10 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v10 + 36) - 238) <= 7
            && ((1LL << (*(_BYTE *)(v10 + 36) + 18)) & 0xAD) != 0 )
          {
            v15 = *(_QWORD *)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
            v16 = *(_DWORD *)(v15 + 32);
            if ( v16 <= 0x40 )
            {
              if ( !*(_QWORD *)(v15 + 24) )
                return v3;
            }
            else if ( v16 == (unsigned int)sub_C444A0(v15 + 24) )
            {
              return v3;
            }
          }
        }
        else
        {
          if ( !v10 || *(_BYTE *)v10 )
            return 1;
          v24 = *(_QWORD *)(v10 + 24);
        }
        if ( v24 == *((_QWORD *)a2 + 10) )
        {
          v12 = *(_DWORD *)(v9 + 16);
          v25[0] = v10;
          if ( v12 )
          {
            v17 = *(unsigned int *)(v9 + 24);
            v18 = *(_QWORD *)(v9 + 8);
            if ( (_DWORD)v17 )
            {
              v19 = (v17 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
              v20 = (__int64 *)(v18 + 8LL * v19);
              v21 = *v20;
              if ( v10 == *v20 )
              {
LABEL_39:
                if ( v20 != (__int64 *)(v18 + 8 * v17) )
                  return 0;
              }
              else
              {
                v22 = 1;
                while ( v21 != -4096 )
                {
                  v23 = v22 + 1;
                  v19 = (v17 - 1) & (v22 + v19);
                  v20 = (__int64 *)(v18 + 8LL * v19);
                  v21 = *v20;
                  if ( v10 == *v20 )
                    goto LABEL_39;
                  v22 = v23;
                }
              }
            }
          }
          else
          {
            v13 = *(_QWORD **)(v9 + 32);
            v14 = &v13[*(unsigned int *)(v9 + 40)];
            if ( v14 != sub_25BD100(v13, (__int64)v14, v25) )
              return 0;
          }
        }
        return 1;
      }
    }
  }
  return v3;
}
