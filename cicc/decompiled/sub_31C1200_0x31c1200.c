// Function: sub_31C1200
// Address: 0x31c1200
//
char __fastcall sub_31C1200(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r8
  _BYTE *v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  int v15; // edx
  int v16; // r9d
  _QWORD v18[12]; // [rsp+0h] [rbp-1A0h] BYREF
  _QWORD v19[12]; // [rsp+60h] [rbp-140h] BYREF
  _QWORD v20[12]; // [rsp+C0h] [rbp-E0h] BYREF
  _QWORD v21[16]; // [rsp+120h] [rbp-80h] BYREF

  v3 = *(unsigned int *)(a1 + 64);
  v4 = *(_QWORD *)(a1 + 48);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      v3 = v4 + 16 * v3;
      if ( v6 != (__int64 *)v3 )
      {
        v8 = (_BYTE *)v6[1];
        if ( v8 )
        {
          if ( !*(_BYTE *)(a1 + 208) || *(_QWORD *)(a1 + 184) == *(_QWORD *)(sub_318B4F0(a2) + 16) + 48LL )
            goto LABEL_6;
          if ( !*(_BYTE *)(a1 + 208) )
            abort();
          v14 = sub_371B3B0(a1 + 176, *(_QWORD *)(a1 + 184), *(_QWORD *)(a1 + 192));
          LOBYTE(v3) = sub_B445A0(*(_QWORD *)(v14 + 16), *(_QWORD *)(a2 + 16));
          if ( (_BYTE)v3 )
          {
            v8[24] = 1;
          }
          else
          {
LABEL_6:
            (*(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(*(_QWORD *)v8 + 24LL))(v21, v8, a1 + 40);
            (*(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(*(_QWORD *)v8 + 16LL))(v20, v8, a1 + 40);
            v18[0] = v20[0];
            v18[1] = v20[1];
            v18[2] = v20[2];
            v18[3] = v20[3];
            v18[4] = v20[4];
            v18[5] = v20[5];
            v18[6] = v20[6];
            v18[7] = v20[7];
            v18[8] = v20[8];
            v18[9] = v20[9];
            v18[10] = v20[10];
            v18[11] = v20[11];
            v19[0] = v21[0];
            v19[1] = v21[1];
            v19[2] = v21[2];
            v19[3] = v21[3];
            v19[4] = v21[4];
            v19[5] = v21[5];
            v19[6] = v21[6];
            v19[7] = v21[7];
            v19[8] = v21[8];
            v19[9] = v21[9];
            v19[10] = v21[10];
            v19[11] = v21[11];
            while ( 1 )
            {
              LOBYTE(v3) = sub_31B8DE0(v18, v19);
              if ( (_BYTE)v3 )
                break;
              v9 = sub_31B8B80((__int64)v18);
              sub_31C1020((__int64 *)a1, v9, v10, v11, v12, v13);
              ++*(_DWORD *)(v9 + 20);
              sub_31B8D10((__int64)v18);
            }
          }
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v7 != -4096 )
      {
        v16 = v15 + 1;
        v5 = (v3 - 1) & (v15 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v15 = v16;
      }
    }
  }
  return v3;
}
