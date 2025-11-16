// Function: sub_153BF40
// Address: 0x153bf40
//
void __fastcall sub_153BF40(_QWORD *a1, __int64 a2, unsigned __int8 a3, __int64 a4, char a5, __m128i *a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // ecx
  size_t v10; // rdx
  char *v11; // r8
  char *v12; // r11
  int v13; // eax
  int v14; // edx
  __int64 v15; // rcx
  int v16; // eax
  _BYTE *v17; // rcx
  unsigned int v18; // edx
  char *v19; // rax
  size_t v20; // r10
  unsigned __int64 v21; // rdx
  unsigned int v22; // r11d
  unsigned int v23; // r11d
  unsigned int v24; // eax
  __int64 v25; // rcx
  char *v26; // r8
  char *v27; // [rsp+0h] [rbp-180h]
  size_t v28; // [rsp+8h] [rbp-178h]
  size_t v29; // [rsp+8h] [rbp-178h]
  size_t v30; // [rsp+8h] [rbp-178h]
  size_t v31; // [rsp+8h] [rbp-178h]
  char *v32; // [rsp+10h] [rbp-170h]
  int v33; // [rsp+10h] [rbp-170h]
  char *v34; // [rsp+10h] [rbp-170h]
  char *v35; // [rsp+10h] [rbp-170h]
  __int64 v39; // [rsp+28h] [rbp-158h]
  void *src; // [rsp+30h] [rbp-150h] BYREF
  size_t n; // [rsp+38h] [rbp-148h]
  _QWORD v42[2]; // [rsp+40h] [rbp-140h] BYREF
  __int64 v43; // [rsp+50h] [rbp-130h] BYREF
  int v44; // [rsp+60h] [rbp-120h]
  unsigned int v45; // [rsp+6Ch] [rbp-114h]
  int v46; // [rsp+74h] [rbp-10Ch]
  __int64 v47[2]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v48; // [rsp+90h] [rbp-F0h]

  v39 = a2;
  src = v42;
  n = 0;
  sub_16CD150(&src, v42, 0x40000, 1);
  v48 = 260;
  v47[0] = (__int64)(a1 + 30);
  sub_16E1010(v42);
  if ( v45 <= 0x1E && (v7 = 1610614920, _bittest64(&v7, v45)) || v46 == 3 )
  {
    v8 = (unsigned int)n;
    v9 = n;
    v10 = (unsigned int)n;
    if ( (_DWORD)n )
    {
      if ( (unsigned __int64)(unsigned int)n + 20 > HIDWORD(n) )
      {
        sub_16CD150(&src, v42, (unsigned int)n + 20LL, 1);
        v10 = (unsigned int)n;
        v9 = n;
      }
      v11 = (char *)src;
      v12 = (char *)src + v10;
      if ( v10 > 0x13 )
      {
        v20 = v10 - 20;
        if ( HIDWORD(n) - v10 < 0x14 )
        {
          v30 = v10 - 20;
          v34 = (char *)src;
          sub_16CD150(&src, v42, v10 + 20, 1);
          v20 = v30;
          v11 = v34;
          v12 = (char *)src + (unsigned int)n;
        }
        v31 = v20;
        v35 = v11;
        memmove(v12, &v11[v20], 0x14u);
        v26 = v35;
        LODWORD(n) = n + 20;
        if ( v31 )
        {
          memmove(v35 + 20, v35, v31);
          v26 = v35;
        }
        *((_DWORD *)v26 + 4) = 0;
        *(_OWORD *)v26 = 0;
      }
      else
      {
        LODWORD(n) = v9 + 20;
        if ( src != v12 )
        {
          v27 = (char *)src + v10;
          v28 = v10;
          v32 = (char *)src;
          memcpy((char *)src + 20, src, v10);
          v12 = v27;
          v10 = v28;
          v11 = v32;
        }
        if ( v10 )
        {
          v29 = (size_t)v12;
          v33 = v10;
          memset(v11, 0, v10);
          LODWORD(v10) = v33;
          v12 = (char *)v29;
        }
        v13 = 20 - v10;
        if ( (unsigned int)(20 - v10) >= 8 )
        {
          *(_QWORD *)v12 = 0;
          *(_QWORD *)&v12[v13 - 8] = 0;
          v21 = (unsigned __int64)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          v22 = (v13 + (_DWORD)v12 - v21) & 0xFFFFFFF8;
          if ( v22 >= 8 )
          {
            v23 = v22 & 0xFFFFFFF8;
            v24 = 0;
            do
            {
              v25 = v24;
              v24 += 8;
              *(_QWORD *)(v21 + v25) = 0;
            }
            while ( v24 < v23 );
          }
        }
        else if ( (v13 & 4) != 0 )
        {
          *(_DWORD *)v12 = 0;
          *(_DWORD *)&v12[v13 - 4] = 0;
        }
        else if ( v13 )
        {
          *v12 = 0;
          if ( (v13 & 2) != 0 )
            *(_WORD *)&v12[v13 - 2] = 0;
        }
      }
    }
    else
    {
      if ( HIDWORD(n) <= 0x13 )
      {
        sub_16CD150(&src, v42, 20, 1);
        v8 = (unsigned int)n;
      }
      v19 = (char *)src + v8;
      *(_OWORD *)v19 = 0;
      *((_DWORD *)v19 + 4) = 0;
      LODWORD(n) = n + 20;
    }
  }
  sub_1525950((__int64)v47, (__int64)&src);
  sub_1538EC0(v47, a1, a3, a4, a5, a6);
  sub_1530C90((__int64)v47);
  sub_152B5E0((__int64)v47);
  v14 = n;
  if ( v45 <= 0x1E && (v15 = 1610614920, _bittest64(&v15, v45)) )
  {
    v16 = v44;
    v17 = &algn_1000005[2];
    if ( v44 != 32 )
    {
LABEL_17:
      LODWORD(v17) = 7;
      if ( v16 != 31 )
      {
        LODWORD(v17) = 18;
        if ( v16 != 16 )
        {
          LODWORD(v17) = 16777234;
          if ( v16 != 17 )
          {
            if ( v16 == 1 || (LODWORD(v17) = -1, v16 == 29) )
              LODWORD(v17) = 12;
          }
        }
      }
    }
  }
  else
  {
    if ( v46 != 3 )
      goto LABEL_26;
    v16 = v44;
    v17 = &algn_1000005[2];
    if ( v44 != 32 )
      goto LABEL_17;
  }
  *(_DWORD *)src = 186106078;
  *((_DWORD *)src + 1) = 0;
  *((_DWORD *)src + 2) = 20;
  *((_DWORD *)src + 3) = v14 - 20;
  *((_DWORD *)src + 4) = (_DWORD)v17;
  v18 = n;
  if ( (n & 0xF) != 0 )
  {
    do
    {
      if ( v18 >= HIDWORD(n) )
        sub_16CD150(&src, v42, 0, 1);
      *((_BYTE *)src + (unsigned int)n) = 0;
      v18 = n + 1;
      LODWORD(n) = v18;
    }
    while ( (v18 & 0xF) != 0 );
  }
LABEL_26:
  sub_16E7EE0(v39, (const char *)src);
  sub_1526880((__int64)v47);
  if ( (__int64 *)v42[0] != &v43 )
    j_j___libc_free_0(v42[0], v43 + 1);
  if ( src != v42 )
    _libc_free((unsigned __int64)src);
}
