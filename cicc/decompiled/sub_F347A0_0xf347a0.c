// Function: sub_F347A0
// Address: 0xf347a0
//
__int64 __fastcall sub_F347A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned int v7; // ebx
  __int64 v8; // rdx
  char *v9; // rax
  unsigned int v10; // r12d
  char v12; // r8
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-80h] BYREF
  char *v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  int v18; // [rsp+18h] [rbp-68h]
  unsigned __int8 v19; // [rsp+1Ch] [rbp-64h]
  char v20; // [rsp+20h] [rbp-60h] BYREF

  v15 = 0;
  v16 = &v20;
  v17 = 8;
  v18 = 0;
  v19 = 1;
  if ( a1 )
  {
    v6 = a1;
    v7 = 0;
    v8 = 1;
    while ( 1 )
    {
      if ( (unsigned int)qword_4F8BE48 <= v7 )
      {
LABEL_18:
        v10 = 0;
        goto LABEL_19;
      }
      if ( !(_BYTE)v8 )
        goto LABEL_11;
      v9 = v16;
      a4 = HIDWORD(v17);
      v8 = (__int64)&v16[8 * HIDWORD(v17)];
      if ( v16 != (char *)v8 )
      {
        while ( *(_QWORD *)v9 != v6 )
        {
          v9 += 8;
          if ( (char *)v8 == v9 )
            goto LABEL_21;
        }
        return 0;
      }
LABEL_21:
      if ( HIDWORD(v17) < (unsigned int)v17 )
      {
        ++HIDWORD(v17);
        *(_QWORD *)v8 = v6;
        LOBYTE(v8) = v19;
        ++v15;
      }
      else
      {
LABEL_11:
        a2 = v6;
        sub_C8CC70((__int64)&v15, v6, v8, a4, a5, a6);
        v12 = v8;
        LOBYTE(v8) = v19;
        if ( !v12 )
          goto LABEL_18;
      }
      v13 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v6 + 48 )
        goto LABEL_26;
      if ( !v13 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_26:
        BUG();
      if ( *(_BYTE *)(v13 - 24) == 36 )
        break;
      if ( sub_AA4F10(v6) )
      {
        LOBYTE(v8) = v19;
        v10 = 1;
        goto LABEL_19;
      }
      ++v7;
      v14 = sub_AA5780(v6);
      v8 = v19;
      v6 = v14;
      if ( !v14 )
        goto LABEL_18;
    }
    v10 = 1;
LABEL_19:
    if ( (_BYTE)v8 )
      return v10;
    _libc_free(v16, a2);
    return v10;
  }
  else
  {
    return 0;
  }
}
