// Function: sub_39FAD60
// Address: 0x39fad60
//
int __fastcall sub_39FAD60(char *filename, struct stat *stat_buf)
{
  return __xstat(1, filename, stat_buf);
}
