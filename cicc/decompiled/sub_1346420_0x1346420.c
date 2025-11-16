// Function: sub_1346420
// Address: 0x1346420
//
__int64 __fastcall sub_1346420(_BYTE *a1, __int64 a2, void *a3, signed __int64 a4, __int64 a5, _BYTE *a6, _BYTE *a7)
{
  unsigned __int64 *v8; // r13
  unsigned int v9; // r14d
  signed __int8 v10; // al
  unsigned __int64 v11; // rax
  void *v12; // rax
  void *v13; // rax
  void *v14; // rbx
  unsigned __int64 v15; // r15
  __int64 v16; // rcx
  void *v17; // r14
  signed __int64 v18; // r8
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r13
  unsigned int *v23; // rax
  unsigned int *v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  char v28; // al
  __int64 v30; // [rsp+18h] [rbp-F8h]
  unsigned __int8 v34; // [rsp+47h] [rbp-C9h]
  unsigned __int64 v35; // [rsp+48h] [rbp-C8h]
  unsigned int i; // [rsp+5Ch] [rbp-B4h]
  _QWORD v37[22]; // [rsp+60h] [rbp-B0h] BYREF

  if ( a4 >= 0 )
  {
    v30 = a2 + 78744;
    v8 = sub_1340A00(a1, a2 + 78744);
    if ( v8 )
    {
      v9 = 0;
      while ( 1 )
      {
        LOBYTE(v37[0]) = 0;
        v10 = _InterlockedCompareExchange8(&byte_4F96C39, 1, 0);
        if ( !v10 )
          break;
        LOBYTE(v37[0]) = v10;
        if ( v9 > 4 )
        {
          sched_yield();
        }
        else
        {
          for ( i = 0; i < 1 << v9; ++i )
            _mm_pause();
          ++v9;
        }
      }
      if ( !byte_4F96C38 )
      {
        while ( 1 )
        {
          v13 = sbrk(0);
          v14 = v13;
          if ( v13 == (void *)-1LL )
            break;
          qword_4F96C30 = (__int64)v13;
          if ( a3 != v13 && a3 != 0 || !v13 )
            break;
          v15 = ((unsigned __int64)v13 + 4095) & 0xFFFFFFFFFFFFF000LL;
          v34 = unk_4C6F2C8;
          v16 = -a5 & (v15 + a5 - 1);
          v17 = (void *)v16;
          v35 = v16 - v15;
          if ( v16 != v15 )
          {
            v18 = sub_13441B0(a2 + 10672);
            v19 = *v8;
            v20 = *(unsigned int *)(a2 + 78928);
            v8[1] = v15;
            v8[4] = v18;
            v8[2] = v35 | v8[2] & 0xFFF;
            *v8 = ((unsigned __int64)v34 << 44) | v20 & 0xFFFFEFFFF0000FFFLL | v19 & 0xFFFFEFFFF0000000LL | 0xE802000;
          }
          v11 = (unsigned __int64)v17 + a4;
          if ( !__CFADD__(a4, v17) )
            v11 = -a5 & (v15 + a5 - 1);
          if ( v11 < (unsigned __int64)v14 )
            break;
          v12 = sbrk((intptr_t)v17 + a4 - (_QWORD)v14);
          if ( v12 == v14 )
          {
            qword_4F96C30 = (__int64)v17 + a4;
            byte_4F96C39 = 0;
            if ( v35 )
            {
              v23 = (unsigned int *)sub_1316370(a2);
              sub_13453D0(a1, a2 + 10672, v23, v8);
            }
            else
            {
              sub_1340AC0((__int64)a1, v30, v8);
            }
            v21 = -a5 & (v15 + a5 - 1);
            if ( *a7 )
            {
              if ( !*a6 )
                return v21;
            }
            else
            {
              v28 = sub_130CC10(v17, a4);
              *a7 = v28;
              if ( !*a6 || !v28 )
                return v21;
            }
            memset(v37, 0, 0x80u);
            v24 = (unsigned int *)sub_1316370(a2);
            v25 = *(unsigned int *)(a2 + 78928);
            v37[1] = -a5 & (v15 + a5 - 1);
            v37[4] = 232;
            v26 = v25 | v37[0] & 0xFFFFFFFFFFFFF000LL;
            v37[2] = a4 | v37[2] & 0xFFF;
            BYTE1(v26) &= ~0x10u;
            v27 = ((unsigned __int64)v34 << 44) | (v26 | ((unsigned __int64)(a4 != 0) << 12)) & 0xFFFFEFFFF0001FFFLL;
            BYTE1(v27) = ((unsigned __int16)((v26 | ((a4 != 0) << 12)) & 0x1FFF) >> 8) | 0x20;
            v37[0] = v27;
            if ( sub_13455F0(a1, v24, (__int64)v37, 0, a4) )
              memset(v17, 0, a4);
            return v21;
          }
          if ( v12 == (void *)-1LL )
          {
            byte_4F96C38 = 1;
            break;
          }
        }
      }
      byte_4F96C39 = 0;
      sub_1340AC0((__int64)a1, v30, v8);
    }
  }
  return 0;
}
