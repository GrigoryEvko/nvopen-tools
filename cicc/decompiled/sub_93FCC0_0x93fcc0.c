// Function: sub_93FCC0
// Address: 0x93fcc0
//
void __fastcall sub_93FCC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  int v5; // edx
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  __int16 v14; // r15
  __int16 v15; // r14
  unsigned __int64 v16; // rsi
  int v17; // [rsp-44h] [rbp-44h] BYREF
  _QWORD v18[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 448) && *(_WORD *)(a1 + 452) )
  {
    sub_B33910(v18);
    if ( !v18[0] )
      goto LABEL_6;
    if ( *(_DWORD *)(a1 + 448) != *(_DWORD *)(a1 + 456) )
    {
      sub_B91220(v18);
      goto LABEL_6;
    }
    v14 = *(_WORD *)(a1 + 452);
    v15 = *(_WORD *)(a1 + 460);
    sub_B91220(v18);
    if ( v14 != v15 )
    {
LABEL_6:
      *(_DWORD *)(a1 + 456) = *(_DWORD *)(a1 + 448);
      *(_WORD *)(a1 + 460) = *(_WORD *)(a1 + 452);
      v3 = *(_QWORD *)(a1 + 512);
      if ( v3 == *(_QWORD *)(a1 + 520) )
        v3 = *(_QWORD *)(*(_QWORD *)(a1 + 536) - 8LL) + 512LL;
      v4 = *(_QWORD *)(v3 - 8);
      sub_93ED80(*(_DWORD *)(a1 + 448), (char *)&v17);
      v5 = *(unsigned __int16 *)(a1 + 452);
      v6 = (_QWORD *)(*(_QWORD *)(v4 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(v4 + 8) & 4) != 0 )
        v6 = (_QWORD *)*v6;
      v7 = sub_B01860((_DWORD)v6, v17, v5, v4, 0, 0, 0, 1);
      sub_B10CB0(v18, v7);
      v8 = v18[0];
      if ( v18[0] )
      {
        v9 = *(unsigned int *)(a2 + 8);
        v10 = *(_QWORD **)a2;
        v11 = *(_DWORD *)(a2 + 8);
        v12 = (_QWORD *)(*(_QWORD *)a2 + 16 * v9);
        if ( *(_QWORD **)a2 != v12 )
        {
          while ( *(_DWORD *)v10 )
          {
            v10 += 2;
            if ( v12 == v10 )
              goto LABEL_19;
          }
          v10[1] = v18[0];
          goto LABEL_16;
        }
LABEL_19:
        v13 = *(unsigned int *)(a2 + 12);
        if ( v9 >= v13 )
        {
          v16 = v9 + 1;
          if ( v13 < v16 )
          {
            sub_C8D5F0(a2, a2 + 16, v16, 16);
            v12 = (_QWORD *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
          }
          *v12 = 0;
          v12[1] = v8;
          v8 = v18[0];
          ++*(_DWORD *)(a2 + 8);
        }
        else
        {
          if ( v12 )
          {
            *(_DWORD *)v12 = 0;
            v12[1] = v8;
            v11 = *(_DWORD *)(a2 + 8);
            v8 = v18[0];
          }
          *(_DWORD *)(a2 + 8) = v11 + 1;
        }
      }
      else
      {
        sub_93FB40(a2, 0);
        v8 = v18[0];
      }
      if ( !v8 )
        return;
LABEL_16:
      sub_B91220(v18);
    }
  }
}
