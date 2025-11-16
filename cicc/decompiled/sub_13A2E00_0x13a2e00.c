// Function: sub_13A2E00
// Address: 0x13a2e00
//
void __fastcall sub_13A2E00(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r15
  _QWORD *v4; // rbx
  __m128i *v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned int v10; // edx
  int v11; // [rsp+Ch] [rbp-64h]
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+30h] [rbp-40h]

  sub_13A29A0(a1);
  if ( *(_DWORD *)(a1 + 344) )
  {
    v2 = *(_QWORD **)(a1 + 336);
    v3 = &v2[3 * *(unsigned int *)(a1 + 352)];
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v4 = v2;
        if ( *v2 != -8 && *v2 != -16 )
          break;
        v2 += 3;
        if ( v3 == v2 )
          return;
      }
      if ( v3 != v2 )
      {
        do
        {
          v5 = *(__m128i **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 <= 0xFu )
          {
            v6 = sub_16E7EE0(a2, "DemandedBits: 0x", 16);
          }
          else
          {
            v6 = a2;
            *v5 = _mm_load_si128((const __m128i *)&xmmword_3F70B10);
            *(_QWORD *)(a2 + 24) += 16LL;
          }
          if ( *((_DWORD *)v4 + 4) > 0x40u )
          {
            v11 = *((_DWORD *)v4 + 4);
            v10 = v11 - sub_16A57B0(v4 + 1);
            v7 = -1;
            if ( v10 <= 0x40 )
              v7 = *(_QWORD *)v4[1];
          }
          else
          {
            v7 = v4[1];
          }
          v12 = v7;
          v13[1] = 0;
          v13[0] = &v12;
          v14 = 271;
          sub_16E2CE0(v13, v6);
          v8 = *(_QWORD *)(v6 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v8) <= 4 )
          {
            v6 = sub_16E7EE0(v6, " for ", 5);
          }
          else
          {
            *(_DWORD *)v8 = 1919903264;
            *(_BYTE *)(v8 + 4) = 32;
            *(_QWORD *)(v6 + 24) += 5LL;
          }
          sub_155C2B0(*v4, v6, 0);
          v9 = *(_BYTE **)(v6 + 24);
          if ( (unsigned __int64)v9 >= *(_QWORD *)(v6 + 16) )
          {
            sub_16E7DE0(v6, 10);
          }
          else
          {
            *(_QWORD *)(v6 + 24) = v9 + 1;
            *v9 = 10;
          }
          v4 += 3;
          if ( v4 == v3 )
            break;
          while ( *v4 == -16 || *v4 == -8 )
          {
            v4 += 3;
            if ( v3 == v4 )
              return;
          }
        }
        while ( v3 != v4 );
      }
    }
  }
}
