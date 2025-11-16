// Function: sub_1B24B00
// Address: 0x1b24b00
//
__int64 __fastcall sub_1B24B00(__int64 a1, __int64 *a2, _QWORD *a3, _BYTE *a4)
{
  unsigned int v4; // r14d
  unsigned int v6; // edx
  _QWORD *v8; // r11
  __int64 v9; // rax
  unsigned int v10; // esi
  unsigned __int64 v11; // rdi
  __int64 v12; // r8
  int v13; // r15d
  __int64 v14; // r10
  unsigned __int64 v15; // r9
  __int64 i; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  float v24; // xmm0_4

  v4 = 0;
  v6 = (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1;
  if ( v6 != 1 )
  {
    v8 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
       ? *(_QWORD **)(a1 - 8)
       : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v4 = 0;
    if ( *(_DWORD *)(*(_QWORD *)*v8 + 8LL) >> 8 != 1 )
    {
      v9 = v8[6];
      v10 = *(_DWORD *)(v9 + 32);
      v11 = *(_QWORD *)(v9 + 24);
      if ( v10 <= 0x40 )
      {
        v12 = (__int64)(v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
      }
      else
      {
        v11 = *(_QWORD *)v11;
        v12 = v11;
      }
      v13 = v6 - 1;
      if ( v6 == 2 )
      {
        v14 = v12;
        v15 = v11;
      }
      else
      {
        v14 = v12;
        v15 = v11;
        for ( i = 2; ; ++i )
        {
          v19 = v8[3 * (unsigned int)(2 * i)];
          v20 = *(_DWORD *)(v19 + 32);
          v17 = *(_QWORD *)(v19 + 24);
          if ( v20 > 0x40 )
          {
            v17 = *(_QWORD *)v17;
            v18 = v17;
          }
          else
          {
            v18 = (__int64)(v17 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
          }
          if ( v11 > v17 )
            v11 = v17;
          if ( v15 < v17 )
            v15 = v17;
          if ( v12 > v18 )
            v12 = v18;
          if ( v14 < v18 )
            v14 = v18;
          if ( v13 == i )
            break;
        }
      }
      v21 = v15 + 1 - v11;
      v22 = v14 + 1 - v12;
      v23 = v22;
      if ( v21 <= v22 )
        v23 = v15 + 1 - v11;
      v4 = 0;
      if ( v23 != 1 )
      {
        v24 = (float)v13;
        if ( v23 < 0 )
        {
          v4 = 0;
          if ( (float)(v24
                     / (float)((float)(v23 & 1 | (unsigned int)((unsigned __int64)v23 >> 1))
                             + (float)(v23 & 1 | (unsigned int)((unsigned __int64)v23 >> 1)))) <= 0.5 )
            return v4;
        }
        else
        {
          v4 = 0;
          if ( (float)(v24 / (float)(int)v23) <= 0.5 )
            return v4;
        }
        if ( v21 <= v22 )
        {
          *a2 = v11;
          v4 = 1;
          *a3 = v15;
          *a4 = 0;
        }
        else
        {
          *a2 = v12;
          v4 = 1;
          *a3 = v14;
          *a4 = 1;
        }
      }
    }
  }
  return v4;
}
