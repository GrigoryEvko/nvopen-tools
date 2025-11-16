// Function: sub_2B203A0
// Address: 0x2b203a0
//
__int64 __fastcall sub_2B203A0(_QWORD *a1, __int64 a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r14
  int v11; // edx
  int *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rsi
  int *v15; // rax
  int *v16; // rcx
  _DWORD *v17; // rcx
  __int64 v18; // rdx
  size_t v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // edx
  int v26; // r8d
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // eax
  __int64 v35; // rcx
  bool v36; // cc
  __int64 v37; // [rsp+8h] [rbp-78h]
  void *s; // [rsp+10h] [rbp-70h] BYREF
  __int64 v39; // [rsp+18h] [rbp-68h]
  _DWORD dest[24]; // [rsp+20h] [rbp-60h] BYREF

  v10 = *(unsigned int *)(a2 + 120);
  if ( !(_DWORD)v10 )
    v10 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)a4 != (_DWORD)v10 )
  {
    v11 = a4;
    v12 = &a3[a4];
    v13 = (4 * a4) >> 4;
    v14 = (4 * a4) >> 2;
    if ( v13 > 0 )
    {
      v15 = a3;
      v16 = &a3[4 * v13];
      while ( v11 > *v15 )
      {
        if ( v11 <= v15[1] )
        {
          if ( v12 != v15 + 1 )
            goto LABEL_12;
          goto LABEL_32;
        }
        if ( v11 <= v15[2] )
        {
          if ( v12 != v15 + 2 )
            goto LABEL_12;
          goto LABEL_32;
        }
        if ( v11 <= v15[3] )
        {
          if ( v12 != v15 + 3 )
            goto LABEL_12;
          goto LABEL_32;
        }
        v15 += 4;
        if ( v16 == v15 )
        {
          v14 = v12 - v15;
          goto LABEL_29;
        }
      }
LABEL_11:
      if ( v12 != v15 )
        goto LABEL_12;
      goto LABEL_32;
    }
    v15 = a3;
LABEL_29:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_32;
        goto LABEL_51;
      }
      if ( (int)a4 <= *v15 )
        goto LABEL_11;
      ++v15;
    }
    if ( (int)a4 <= *v15 )
      goto LABEL_11;
    ++v15;
LABEL_51:
    if ( (int)a4 <= *v15 )
      goto LABEL_11;
LABEL_32:
    if ( (unsigned __int8)sub_B4ED80(a3, a4, a4) )
      return a2;
LABEL_12:
    s = dest;
    v39 = 0xC00000000LL;
    if ( (unsigned int)v10 > 0xC )
    {
      sub_C8D5F0((__int64)&s, dest, (unsigned int)v10, 4u, a5, a6);
      memset(s, 255, 4LL * (unsigned int)v10);
      LODWORD(v39) = v10;
      v17 = s;
    }
    else
    {
      if ( (_DWORD)v10 )
      {
        v30 = 4LL * (unsigned int)v10;
        if ( v30 )
        {
          if ( (unsigned int)v30 >= 8 )
          {
            v31 = (unsigned int)v30;
            v32 = v30 - 1;
            *(_QWORD *)((char *)&dest[-2] + v31) = -1;
            if ( v32 >= 8 )
            {
              v33 = v32 & 0xFFFFFFF8;
              v34 = 0;
              do
              {
                v35 = v34;
                v34 += 8;
                *(_QWORD *)((char *)dest + v35) = -1;
              }
              while ( v34 < v33 );
            }
          }
          else if ( ((4 * (_BYTE)v10) & 4) != 0 )
          {
            dest[0] = -1;
            *(_DWORD *)((char *)&dest[-1] + (unsigned int)v30) = -1;
          }
          else if ( (_DWORD)v30 )
          {
            LOBYTE(dest[0]) = -1;
          }
        }
      }
      LODWORD(v39) = v10;
      v17 = dest;
    }
    v18 = (unsigned int)a4;
    if ( (unsigned int)a4 > (unsigned int)v10 )
      v18 = v10;
    v19 = 4 * v18;
    if ( v19 )
    {
      memmove(v17, a3, v19);
      v17 = s;
    }
    v20 = (unsigned int)v39;
    v21 = *(_QWORD *)(*(_QWORD *)(a2 + 416) + 8LL);
    v22 = *(unsigned __int8 *)(v21 + 8);
    if ( (_BYTE)v22 == 17 )
    {
      LODWORD(v10) = *(_DWORD *)(v21 + 32) * v10;
    }
    else if ( (unsigned int)(v22 - 17) > 1 )
    {
      goto LABEL_22;
    }
    v21 = **(_QWORD **)(v21 + 16);
LABEL_22:
    v37 = (__int64)v17;
    v23 = sub_BCDA70((__int64 *)v21, v10);
    v24 = sub_DFBC30(*(__int64 **)(*a1 + 3296LL), 7, v23, v37, v20, 0, 0, 0, 0, 0, 0);
    v26 = v25;
    v27 = a1[1];
    if ( v26 == 1 )
      *(_DWORD *)(v27 + 8) = 1;
    if ( __OFADD__(*(_QWORD *)v27, v24) )
    {
      v36 = v24 <= 0;
      v28 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v36 )
        v28 = 0x8000000000000000LL;
    }
    else
    {
      v28 = *(_QWORD *)v27 + v24;
    }
    *(_QWORD *)v27 = v28;
    if ( s != dest )
      _libc_free((unsigned __int64)s);
  }
  return a2;
}
