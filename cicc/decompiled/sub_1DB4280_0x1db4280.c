// Function: sub_1DB4280
// Address: 0x1db4280
//
void __fastcall sub_1DB4280(__int64 *a1)
{
  _BYTE *v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  _BYTE *v6; // r8
  int v7; // r9d
  char v8; // dl
  __int64 v9; // rax
  _DWORD *v10; // r13
  _QWORD *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE *v15; // [rsp+8h] [rbp-98h]
  _BYTE *v16; // [rsp+10h] [rbp-90h]
  __int64 v17; // [rsp+18h] [rbp-88h]
  int v18; // [rsp+20h] [rbp-80h]
  _BYTE v19[120]; // [rsp+28h] [rbp-78h] BYREF

  v2 = v19;
  v3 = *((unsigned int *)a1 + 2);
  v4 = *a1;
  v14 = 0;
  v15 = v19;
  v16 = v19;
  v5 = v4 + 24 * v3;
  v18 = 0;
  v17 = 8;
  *((_DWORD *)a1 + 18) = 0;
  if ( v5 != v4 )
  {
    v6 = v19;
    while ( 1 )
    {
      v10 = *(_DWORD **)(v4 + 16);
      if ( v2 == v6 )
      {
        v11 = &v2[8 * HIDWORD(v17)];
        v7 = HIDWORD(v17);
        if ( v11 != (_QWORD *)v2 )
        {
          v12 = v2;
          v13 = 0;
          while ( v10 != (_DWORD *)*v12 )
          {
            if ( *v12 == -2 )
              v13 = v12;
            if ( v11 == ++v12 )
            {
              if ( !v13 )
                goto LABEL_20;
              *v13 = v10;
              --v18;
              ++v14;
              goto LABEL_4;
            }
          }
          goto LABEL_7;
        }
LABEL_20:
        if ( HIDWORD(v17) < (unsigned int)v17 )
          break;
      }
      sub_16CCBA0((__int64)&v14, *(_QWORD *)(v4 + 16));
      v6 = v16;
      v2 = v15;
      if ( v8 )
        goto LABEL_4;
LABEL_7:
      v4 += 24;
      if ( v4 == v5 )
      {
        if ( v6 != v2 )
          _libc_free((unsigned __int64)v6);
        return;
      }
    }
    v7 = ++HIDWORD(v17);
    *v11 = v10;
    ++v14;
LABEL_4:
    *v10 = *((_DWORD *)a1 + 18);
    v9 = *((unsigned int *)a1 + 18);
    if ( (unsigned int)v9 >= *((_DWORD *)a1 + 19) )
    {
      sub_16CD150((__int64)(a1 + 8), a1 + 10, 0, 8, (int)v6, v7);
      v9 = *((unsigned int *)a1 + 18);
    }
    *(_QWORD *)(a1[8] + 8 * v9) = v10;
    v6 = v16;
    ++*((_DWORD *)a1 + 18);
    v2 = v15;
    goto LABEL_7;
  }
}
