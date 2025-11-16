// Function: sub_A02A70
// Address: 0xa02a70
//
__int64 __fastcall sub_A02A70(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v6; // r15
  _BYTE *v7; // rax
  _BYTE *v8; // r15
  size_t v9; // r8
  __int64 v10; // r13
  _BYTE *v11; // rdi
  int v12; // edx
  _BYTE *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rdx
  size_t v22; // [rsp+0h] [rbp-90h]
  _BYTE *v23; // [rsp+8h] [rbp-88h]
  _BYTE *v24; // [rsp+10h] [rbp-80h] BYREF
  __int64 v25; // [rsp+18h] [rbp-78h]
  _BYTE dest[112]; // [rsp+20h] [rbp-70h] BYREF

  result = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v3 = *(_QWORD *)(result + 24);
  if ( v3 )
  {
    result = sub_AF4730(*(_QWORD *)(result + 24));
    if ( (_BYTE)result )
    {
      result = sub_B58EB0(a2, 0);
      if ( result )
      {
        if ( *(_BYTE *)result == 22 )
        {
          v6 = *(_QWORD *)(v3 + 16);
          v25 = 0x800000000LL;
          v7 = *(_BYTE **)(v3 + 24);
          v8 = (_BYTE *)(v6 + 8);
          v24 = dest;
          v9 = v7 - v8;
          v10 = (v7 - v8) >> 3;
          if ( (unsigned __int64)(v7 - v8) > 0x40 )
          {
            v22 = v7 - v8;
            v23 = v7;
            sub_C8D5F0(&v24, dest, (v7 - v8) >> 3, 8);
            v13 = v24;
            v12 = v25;
            v7 = v23;
            v9 = v22;
            v11 = &v24[8 * (unsigned int)v25];
          }
          else
          {
            v11 = dest;
            v12 = 0;
            v13 = dest;
          }
          if ( v7 != v8 )
          {
            memcpy(v11, v8, v9);
            v13 = v24;
            v12 = v25;
          }
          v14 = *a1;
          LODWORD(v25) = v10 + v12;
          v15 = sub_B0D000(*(_QWORD *)(v14 + 248), v13, (unsigned int)(v10 + v12), 0, 1);
          v16 = *(_QWORD *)(v15 + 8);
          v17 = (_QWORD *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v16 & 4) != 0 )
            v17 = (_QWORD *)*v17;
          result = sub_B9F6F0(v17, v15);
          v18 = 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v19 = v18 + a2;
          if ( *(_QWORD *)v19 )
          {
            v20 = *(_QWORD *)(v19 + 8);
            **(_QWORD **)(v19 + 16) = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = *(_QWORD *)(v19 + 16);
          }
          *(_QWORD *)v19 = result;
          if ( result )
          {
            v21 = *(_QWORD *)(result + 16);
            *(_QWORD *)(v19 + 8) = v21;
            if ( v21 )
            {
              v18 = v19 + 8;
              *(_QWORD *)(v21 + 16) = v19 + 8;
            }
            *(_QWORD *)(v19 + 16) = result + 16;
            *(_QWORD *)(result + 16) = v19;
          }
          if ( v24 != dest )
            return _libc_free(v24, v18);
        }
      }
    }
  }
  return result;
}
