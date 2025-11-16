// Function: sub_20FFD80
// Address: 0x20ffd80
//
__int64 __fastcall sub_20FFD80(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // r14
  __int64 result; // rax
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 *v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rbx
  int v13; // r11d
  unsigned int v14; // eax
  __int64 v15; // rdx
  int v16; // ecx
  __int64 v17; // r8
  __int64 v18; // r10
  unsigned __int64 v19; // rsi
  unsigned int v20; // r13d
  __int64 v21; // rcx
  _QWORD *v22; // rdx
  _QWORD *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  int v28; // [rsp+18h] [rbp-48h]
  _QWORD *v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(__int64 **)(v3 + 64);
  result = *(unsigned int *)(v3 + 72);
  v6 = &v4[result];
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      v11 = *v4;
      v12 = *(_QWORD *)(*v4 + 8);
      if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        break;
LABEL_10:
      if ( v6 == ++v4 )
        goto LABEL_20;
    }
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 112LL);
    v14 = v13 & 0x7FFFFFFF;
    v15 = v13 & 0x7FFFFFFF;
    v16 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 312LL) + 4 * v15);
    if ( v16 )
    {
      v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 312LL) + 4 * v15);
      v14 = v16 & 0x7FFFFFFF;
      v15 = v16 & 0x7FFFFFFF;
    }
    v17 = *(_QWORD *)(a1 + 32);
    v18 = 8 * v15;
    v19 = *(unsigned int *)(v17 + 408);
    if ( (unsigned int)v19 > v14 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(v17 + 400) + 8 * v15);
      if ( v7 )
      {
LABEL_4:
        v8 = (__int64 *)sub_1DB3C70((__int64 *)v7, v12);
        result = *(_QWORD *)v7 + 24LL * *(unsigned int *)(v7 + 8);
        if ( v8 != (__int64 *)result )
        {
          result = *(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3;
          if ( (unsigned int)result <= (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3) )
          {
            v9 = v8[2];
            if ( v9 )
            {
              result = *(_QWORD *)(v9 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              if ( result )
              {
                v10 = *(_QWORD *)(result + 16);
                if ( v10 )
                  result = sub_20FFC70(a1, v9, v10, a2);
              }
            }
          }
        }
        goto LABEL_10;
      }
    }
    v20 = v14 + 1;
    if ( (unsigned int)v19 < v14 + 1 )
    {
      if ( v20 >= v19 )
      {
        if ( v20 > v19 )
        {
          if ( v20 > (unsigned __int64)*(unsigned int *)(v17 + 412) )
          {
            v25 = 8 * v15;
            v28 = v13;
            v30 = *v4;
            v32 = *(_QWORD *)(a1 + 32);
            sub_16CD150(v17 + 400, (const void *)(v17 + 416), v20, 8, v17, v11);
            v17 = v32;
            v18 = v25;
            v13 = v28;
            v11 = v30;
            v19 = *(unsigned int *)(v32 + 408);
          }
          v21 = *(_QWORD *)(v17 + 400);
          v22 = (_QWORD *)(v21 + 8 * v19);
          v23 = (_QWORD *)(v21 + 8LL * v20);
          v24 = *(_QWORD *)(v17 + 416);
          if ( v23 != v22 )
          {
            do
              *v22++ = v24;
            while ( v23 != v22 );
            v21 = *(_QWORD *)(v17 + 400);
          }
          *(_DWORD *)(v17 + 408) = v20;
          goto LABEL_17;
        }
      }
      else
      {
        *(_DWORD *)(v17 + 408) = v20;
      }
    }
    v21 = *(_QWORD *)(v17 + 400);
LABEL_17:
    v27 = v11;
    v29 = (_QWORD *)v17;
    v31 = v18;
    *(_QWORD *)(v21 + v18) = sub_1DBA290(v13);
    v7 = *(_QWORD *)(v29[50] + v31);
    sub_1DBB110(v29, v7);
    v12 = *(_QWORD *)(v27 + 8);
    goto LABEL_4;
  }
LABEL_20:
  *(_BYTE *)(a1 + 68) = 1;
  return result;
}
