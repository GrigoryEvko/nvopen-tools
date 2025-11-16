// Function: sub_A215B0
// Address: 0xa215b0
//
void __fastcall sub_A215B0(__int64 a1, unsigned int a2, char *a3, __int64 a4, unsigned int a5)
{
  char *v7; // r15
  _QWORD *v8; // r11
  char *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // r12d
  __int64 v14; // rsi
  unsigned int v15; // [rsp+Ch] [rbp-154h]
  __int64 v16; // [rsp+10h] [rbp-150h]
  _QWORD *v17; // [rsp+18h] [rbp-148h]
  _BYTE *v18; // [rsp+20h] [rbp-140h] BYREF
  __int64 v19; // [rsp+28h] [rbp-138h]
  _BYTE v20[304]; // [rsp+30h] [rbp-130h] BYREF

  v7 = &a3[a4];
  v8 = &v18;
  v18 = v20;
  v19 = 0x4000000000LL;
  if ( &a3[a4] != a3 )
  {
    v10 = a3;
    v11 = 0;
    v12 = 64;
    v8 = &v18;
    while ( 1 )
    {
      v13 = *v10;
      if ( a5 && (unsigned __int8)((v13 & 0xDF) - 65) > 0x19u )
      {
        if ( (unsigned __int8)(v13 - 46) > 0x31u )
        {
          a5 = 0;
        }
        else if ( ((0x2000000000FFDuLL >> ((unsigned __int8)v13 - 46)) & 1) == 0 )
        {
          a5 = 0;
        }
      }
      if ( v11 + 1 > v12 )
      {
        v15 = a2;
        v16 = a1;
        v17 = v8;
        sub_C8D5F0(v8, v20, v11 + 1, 4);
        v11 = (unsigned int)v19;
        a2 = v15;
        a1 = v16;
        v8 = v17;
      }
      ++v10;
      *(_DWORD *)&v18[4 * v11] = v13;
      v11 = (unsigned int)(v19 + 1);
      LODWORD(v19) = v19 + 1;
      if ( v7 == v10 )
        break;
      v12 = HIDWORD(v19);
    }
  }
  v14 = a2;
  sub_A214F0(a1, a2, (__int64)v8, a5);
  if ( v18 != v20 )
    _libc_free(v18, v14);
}
